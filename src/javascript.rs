use neon::{prelude::*, result::NeonResult, types::Value};
use serde::{de::DeserializeOwned, Serialize};

pub fn from_js<'a, T: DeserializeOwned, V: Value, C: Context<'a>>(
  cx: &mut C,
  v: Handle<'a, V>,
) -> NeonResult<T> {
  let stringify = cx
    .global::<JsObject>("JSON")?
    .get::<JsFunction, _, _>(cx, "stringify")?;

  let json = stringify
    .call(cx, v, [v.upcast()])?
    .downcast_or_throw::<JsString, _>(cx)?
    .value(cx);

  let s = serde_json::from_str(&json).or_else(|e| cx.throw_error(e.to_string()))?;

  Ok(s)
}

pub fn to_js<'a, T: Serialize, C: Context<'a>>(
  cx: &mut C,
  s: T,
) -> NeonResult<Handle<'a, JsValue>> {
  let json = serde_json::to_string(&s).or_else(|e| cx.throw_error(e.to_string()))?;

  let parse = cx
    .global::<JsObject>("JSON")?
    .get::<JsFunction, _, _>(cx, "parse")?;

  let this = cx.string(&json).upcast();

  let parsed = parse
    .call(cx, this, [this])?
    .downcast_or_throw::<JsValue, _>(cx)?;

  Ok(parsed)
}
