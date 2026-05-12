//! reedline `Prompt` impl.
//!
//! Main prompt is the mathematical double-struck k (`𝕜`) in bold cyan,
//! followed by a thin space. Multi-line continuation is a centred dot.

use std::borrow::Cow;

use reedline::{
    Color, Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, PromptViMode,
};

pub struct KPrompt;

impl Prompt for KPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("\u{1D55C} ")
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
        Cow::Borrowed("\u{00B7} ") // middle dot
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

    fn get_prompt_color(&self) -> Color {
        Color::Cyan
    }
    fn get_prompt_multiline_color(&self) -> nu_ansi_term::Color {
        nu_ansi_term::Color::DarkGray
    }
    fn get_indicator_color(&self) -> Color {
        Color::Cyan
    }
    fn get_prompt_right_color(&self) -> Color {
        Color::DarkGrey
    }
}
