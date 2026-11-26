use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

fn array_len(value: Option<&Value>) -> usize {
    value.and_then(Value::as_array).map_or(0, Vec::len)
}

pub fn inspect(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let root: Value = serde_json::from_str(&text).context("parse golden JSON")?;
    let clips = root
        .get("clips")
        .and_then(Value::as_array)
        .context("golden has no clips array")?;

    let mut by_lang = BTreeMap::new();
    let mut words = 0;
    let mut segments = 0;
    let mut token_ids = 0;
    let mut token_durations = 0;
    let mut empty_text = 0;

    for clip in clips {
        let lang = clip
            .get("lang")
            .and_then(Value::as_str)
            .context("clip has no lang")?;
        *by_lang.entry(lang.to_string()).or_insert(0usize) += 1;
        words += array_len(clip.get("words"));
        segments += array_len(clip.get("segments"));
        token_ids += array_len(clip.get("token_ids"));
        token_durations += array_len(clip.get("token_duration"));
        if clip
            .get("text")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            empty_text += 1;
        }
    }

    if clips.is_empty() {
        bail!("golden clips array is empty");
    }

    let summary = json!({
        "path": path.to_string_lossy(),
        "clip_count": clips.len(),
        "by_lang": by_lang,
        "words": words,
        "segments": segments,
        "token_ids": token_ids,
        "token_durations": token_durations,
        "empty_text": empty_text,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
