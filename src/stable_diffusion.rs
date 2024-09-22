use std::path::PathBuf;

use anyhow::Result;
use base64::Engine;
use candle_core::{
  utils::{cuda_is_available, metal_is_available},
  DType, Device, Module, Tensor, D,
};
use candle_transformers::models::stable_diffusion::{
  build_clip_transformer, StableDiffusionConfig,
};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

// Version of Stable Diffusion to use
#[derive(Debug, Default)]
pub enum Version {
  #[default]
  XL,
  Turbo,
  SD2_1,
}

impl From<String> for Version {
  fn from(s: String) -> Self {
    match s.as_str() {
      "xl" => Version::XL,
      "turbo" => Version::Turbo,
      "sd2.1" => Version::SD2_1,
      _ => Version::Turbo,
    }
  }
}

type ModelFile = (Repo, String);

trait ModelFileExt {
  fn get(&self, api: &Api) -> Result<PathBuf>;
}

impl ModelFileExt for ModelFile {
  fn get(&self, api: &Api) -> Result<PathBuf> {
    let (repo, path) = self;
    let repo = api.repo(repo.to_owned());
    let file = repo.get(path)?;

    Ok(file)
  }
}

pub struct Generator {
  pub version: Version,
  pub use_f16: bool,
  pub use_cpu: bool,
}

impl<'a> TryFrom<&'a Generator> for Model {
  type Error = anyhow::Error;

  fn try_from(g: &'a Generator) -> Result<Self> {
    let embedders = match &g.version {
      Version::SD2_1 => vec![Repo::model("openai/clip-vit-base-patch32".to_string())],
      Version::Turbo | Version::XL => {
        vec![
          Repo::model("openai/clip-vit-large-patch14".to_string()),
          Repo::model("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string()),
        ]
      }
    }
    .iter()
    .map(|m| (m.to_owned(), "tokenizer.json".to_string()))
    .collect();

    let model_repo = match &g.version {
      Version::SD2_1 => Repo::model("stabilityai/stable-diffusion-2-1".to_string()),
      Version::XL => Repo::model("stabilityai/stable-diffusion-xl-base-1.0".to_string()),
      Version::Turbo => Repo::model("stabilityai/sdxl-turbo".to_string()),
    };

    let safetensors_ext = match g.use_f16 {
      true => ".fp16.safetensors",
      false => ".safetensors",
    };

    Ok(Model {
      hf_api: Api::new()?,

      guidance_scale: match &g.version {
        Version::XL | Version::SD2_1 => Some(7.5),
        _ => None,
      },
      vae_scale: match &g.version {
        Version::XL | Version::SD2_1 => 0.18215,
        Version::Turbo => 0.13025,
      },
      steps: match &g.version {
        Version::XL | Version::SD2_1 => 30,
        _ => 1,
      },
      use_cpu: g.use_cpu,
      dtype: if g.use_f16 { DType::F16 } else { DType::F32 },
      config: match &g.version {
        Version::XL => StableDiffusionConfig::sdxl(None, Some(1024), Some(1024)),
        Version::Turbo => StableDiffusionConfig::sdxl_turbo(None, Some(1024), Some(1024)),
        Version::SD2_1 => StableDiffusionConfig::v2_1(None, Some(1024), Some(1024)),
      },
      embedders,
      clips: match &g.version {
        Version::SD2_1 => vec![(
          model_repo.clone(),
          format!("text_encoder/model{}", safetensors_ext),
        )],
        Version::XL | Version::Turbo => {
          vec![
            (
              model_repo.clone(),
              format!("text_encoder/model{}", safetensors_ext),
            ),
            (
              model_repo.clone(),
              format!("text_encoder_2/model{}", safetensors_ext),
            ),
          ]
        }
      },
      unet: (
        model_repo.clone(),
        format!("unet/diffusion_pytorch_model{}", safetensors_ext),
      ),
      vae: (
        model_repo.clone(),
        format!("vae/diffusion_pytorch_model{}", safetensors_ext),
      ),
    })
  }
}

impl Generator {
  pub fn generate(&self, prompt: &str, negative_prompt: &str) -> Result<String> {
    let model: Model = self.try_into()?;
    model.generate(prompt, negative_prompt)
  }
}

struct Model {
  hf_api: Api,

  guidance_scale: Option<f64>,
  vae_scale: f64,
  steps: usize,
  use_cpu: bool,
  dtype: DType,
  config: StableDiffusionConfig,

  embedders: Vec<ModelFile>,
  clips: Vec<ModelFile>,
  unet: ModelFile,
  vae: ModelFile,
}

// @agent generate an image of a cat in space

impl Model {
  fn generate(&self, prompt: &str, negative_prompt: &str) -> Result<String> {
    let dev = device(self.use_cpu)?;

    let embeds = self.text_embeds(prompt, negative_prompt, &dev)?;

    let vae_weights = self.vae.get(&self.hf_api)?;
    let vae = self.config.build_vae(vae_weights, &dev, self.dtype)?;

    let unet_weights = self.unet.get(&self.hf_api)?;
    let unet = self
      .config
      .build_unet(unet_weights, &dev, 4, false, self.dtype)?;

    let scheduler = self.config.build_scheduler(self.steps)?;
    let timesteps = scheduler.timesteps();
    let latents = {
      let latents = Tensor::randn(
        0f32,
        1f32,
        (1, 4, self.config.height / 8, self.config.width / 8),
        &dev,
      )?;

      (latents * scheduler.init_noise_sigma())?
    };

    let mut latents = latents.to_dtype(self.dtype)?;

    for &timestep in timesteps.iter() {
      let (latent_input, embeds) = if self.guidance_scale.is_some() {
        (
          Tensor::cat(&[&latents, &latents], 0)?,
          embeds.repeat((1, 1, 1))?,
        )
      } else {
        (latents.clone(), embeds.clone())
      };
      let latent_input = scheduler.scale_model_input(latent_input, timestep)?;

      let noise = unet.forward(&latent_input, timestep as f64, &embeds)?;
      let noise = match self.guidance_scale {
        Some(scale) => {
          let noise = noise.chunk(2, 0)?;
          let (noise_negative, noise_prompt) = (&noise[0], &noise[1]);

          (noise_negative + ((noise_prompt - noise_negative)? * scale)?)?
        }
        None => noise,
      };
      // rank error here

      latents = scheduler.step(&noise, timestep, &latents)?;
    }

    let img = vae.decode(&(latents / self.vae_scale)?)?;
    let img = ((img / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let img = (img.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
      anyhow::bail!("Expected 3 channels, got {}", channel);
    }

    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;

    Ok(base64::engine::general_purpose::STANDARD.encode(&pixels))
  }

  fn text_embeds(&self, prompt: &str, negative_prompt: &str, device: &Device) -> Result<Tensor> {
    let embeddings = self
      .embedders
      .iter()
      .zip(self.clips.iter())
      .enumerate()
      .map(|(pass, (embedder, clip))| {
        let tokens = self.encode_prompt_tensor(prompt, embedder, device)?;

        let model = build_clip_transformer(
          if pass == 0 {
            &self.config.clip
          } else {
            self.config.clip2.as_ref().unwrap()
          },
          clip.get(&self.hf_api)?,
          device,
          DType::F32,
        )?;

        let embeddings = model.forward(&tokens)?;
        let embeddings = match self.guidance_scale {
          Some(_) => {
            let tokens = self.encode_prompt_tensor(negative_prompt, embedder, device)?;
            let negative_embeddings = model.forward(&tokens)?;

            Tensor::cat(&[negative_embeddings, embeddings], 0)?.to_dtype(self.dtype)?
          }
          None => embeddings.to_dtype(self.dtype)?,
        };

        Ok(embeddings)
      })
      .collect::<Result<Vec<_>>>()?;

    let embeddings = Tensor::cat(&embeddings, 0)?;
    let embeddings = embeddings.repeat((1, 1, 1))?;

    Ok(embeddings)
  }

  fn encode_prompt_tensor(
    &self,
    prompt: &str,
    embedder: &ModelFile,
    device: &Device,
  ) -> Result<Tensor> {
    let tokenizer = embedder.get(&self.hf_api)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(anyhow::Error::msg)?;

    // add padding
    let padding = match &self.config.clip.pad_with {
      Some(pad) => *tokenizer.get_vocab(true).get(pad.as_str()).unwrap(),
      None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };

    // encode tokens in promp
    let mut tokens = tokenizer
      .encode(prompt, true)
      .map_err(anyhow::Error::msg)?
      .get_ids()
      .to_vec();

    if tokens.len() > self.config.clip.max_position_embeddings {
      anyhow::bail!(
        "Prompt is too long ({}). Max: {}",
        tokens.len(),
        self.config.clip.max_position_embeddings
      );
    }

    while tokens.len() < self.config.clip.max_position_embeddings {
      tokens.push(padding);
    }

    Tensor::new(tokens.as_slice(), device)?
      .unsqueeze(0)
      .map_err(anyhow::Error::msg)
  }
}

fn device(cpu: bool) -> Result<Device> {
  let d = if cpu {
    Device::Cpu
  } else if cuda_is_available() {
    Device::new_cuda(0)?
  } else if metal_is_available() {
    Device::new_metal(0)?
  } else {
    Device::Cpu
  };

  Ok(d)
}
