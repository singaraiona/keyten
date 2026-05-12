//! reedline `Prompt` impl — minimal `k) ` prompt with subtle colouring.

use std::borrow::Cow;

use reedline::{
    Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, PromptViMode,
};

pub struct KPrompt;

impl Prompt for KPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("k) ")
    }
    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_indicator(&self, mode: PromptEditMode) -> Cow<'_, str> {
        match mode {
            PromptEditMode::Default | PromptEditMode::Emacs => Cow::Borrowed(""),
            PromptEditMode::Vi(PromptViMode::Normal) => Cow::Borrowed("[N] "),
            PromptEditMode::Vi(PromptViMode::Insert) => Cow::Borrowed("[I] "),
            PromptEditMode::Custom(s) => Cow::Owned(format!("({s}) ")),
        }
    }
    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("..   ")
    }
    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let tag = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "(failing) ",
        };
        Cow::Owned(format!("({tag}reverse-i-search: {}) ", history_search.term))
    }
}
