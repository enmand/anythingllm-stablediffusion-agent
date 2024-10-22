mod javascript;
mod stable_diffusion;

use neon::{object::PropertyKey, prelude::*};

struct AnythingLLM<'a> {
  this: Handle<'a, JsObject>,
}

impl<'a> AnythingLLM<'a> {
  fn introspect<'b, C: Context<'b>>(
    &self,
    cx: &mut C,
    log: Handle<'b, impl Value>,
  ) -> JsResult<'b, JsUndefined> {
    let introspect: Handle<JsFunction> = self.this.get(cx, "introspect")?;

    let _: Handle<JsUndefined> = introspect.call_with(cx).arg(log).apply(cx)?;

    Ok(cx.undefined())
  }

  fn log<'b, C: Context<'b>>(
    &self,
    cx: &mut C,
    log: Handle<'b, impl Value>,
  ) -> JsResult<'b, JsUndefined> {
    let log: Handle<JsFunction> = self.this.get(cx, "log")?;

    let _: Handle<JsUndefined> = log.call_with(cx).arg(log).apply(cx)?;

    Ok(cx.undefined())
  }

  fn runtime_arg<'b, V: Value, C: Context<'b>, K: PropertyKey>(
    &self,
    cx: &mut C,
    key: K,
  ) -> JsResult<'b, V> {
    let runtime_args: Handle<JsObject> = self.this.get(cx, "runtimeArgs")?;

    runtime_args.get(cx, key)
  }
}

fn handle(mut cx: FunctionContext) -> JsResult<JsPromise> {
  let plugin: AnythingLLM = AnythingLLM { this: cx.this()? };

  // Get the prompt from the arguments
  let prompt: String = {
    cx.argument::<JsObject>(0)?
      .get::<JsString, _, _>(&mut cx, "prompt")?
      .value(&mut cx)
  };
  // Get the prompt from the arguments
  let negative_prompt: String = cx
    .argument::<JsObject>(0)?
    .get::<JsString, _, _>(&mut cx, "negative_prompt")?
    .value(&mut cx);

  // Get the stable diffusion model argument
  let version = plugin
    .runtime_arg::<JsString, _, _>(&mut cx, "MODEL")?
    .value(&mut cx)
    .into();

  let use_f16 = plugin
    .runtime_arg::<JsString, _, _>(&mut cx, "USE_FP_16")?
    .value(&mut cx)
    .parse::<i32>()
    .or_else(|e| cx.throw_error(format!("USE_FP_16 must be a boolean: {}", e)))?
    > 0;

  let use_cpu = plugin
    .runtime_arg::<JsString, _, _>(&mut cx, "USE_CPU")?
    .value(&mut cx)
    .parse::<i32>()
    .or_else(|_| cx.throw_error("USE_CPU must be a boolean"))?
    > 0;

  // Create the model
  let model = stable_diffusion::Generator {
    version,
    use_f16,
    use_cpu,
  };

  // Log the operation
  let log = JsString::new(
    &mut cx,
    format!(
      "generating image with prompt with {:?} (fp16: {}): {} (negative prompt: {}))\nfirst run has to download model weights, and may take some time",
      model.version, model.use_f16, prompt, negative_prompt
    )
    .as_str(),
  );
  plugin.introspect(&mut cx, log)?;

  let img = model
    .generate(&prompt, &negative_prompt)
    .or_else(|e| cx.throw_error(format!("{}\n{}", e, e.backtrace())))?;

  let (def, prom) = cx.promise();
  let resp = cx.string(format!("You are an AI agent tool that has been called to generate an image, using another model to build the image.

Present only the image generated by the model to the user as markdown, and nothing else.

![](data:image/png;base64,{img})
"));

  def.resolve(&mut cx, resp);

  Ok(prom)
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
  cx.export_function("handler", handle)?;

  Ok(())
}
