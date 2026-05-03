"""Gradio app for AI Incident Analyzer."""

from __future__ import annotations  # Keep modern type hints working consistently.

import re  # Parse the LLM's sectioned answer into separate UI output boxes.

import gradio as gr  # Build the browser-based incident analysis screen.

from .incident_analyzer import analyze_incident  # Reuse the factored analyzer workflow.


APP_CSS = """
body, .gradio-container {
    background: #f3f4f6;
}

.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}

.hero {
    background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 48%, #db2777 100%);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 14px;
    padding: 28px 30px;
    margin-bottom: 18px;
    box-shadow: 0 18px 45px rgba(31, 41, 55, 0.18);
}

.hero h1 {
    color: #ffffff;
    font-size: 34px;
    line-height: 1.2;
    margin: 0 0 8px;
}

.hero p {
    color: #eef2ff;
    font-size: 16px;
    margin: 0;
}

.hero .eyebrow {
    color: #dbeafe;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.panel {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 16px;
}

.input-card {
    min-height: 360px;
}

.repo-card {
    max-width: 260px;
}

.cost-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 16px;
}

.cost-box textarea,
.cost-box input,
.cost-box label,
.cost-box span {
    color: #000000 !important;
    font-weight: 700;
}

.section-label {
    color: #374151;
    font-weight: 700;
    margin: 4px 0 10px;
}

#submit-btn {
    min-height: 44px;
    font-weight: 700;
}

.output-box textarea,
.output-box input,
.output-box label,
.output-box span,
.output-box .prose {
    color: #000000 !important;
}

.output-box textarea {
    background: #ffffff !important;
    border: 1px solid #9ca3af !important;
}
"""


APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
)


def _file_paths(files) -> list[str]:
    """Normalize Gradio upload values to filesystem paths."""

    paths: list[str] = []  # Collect only real paths that the analyzer can read.
    for file in files or []:  # Gradio can return several upload object shapes.
        if isinstance(file, str):  # Newer Gradio versions may return a path string.
            paths.append(file)  # Store the uploaded file path as-is.
        elif hasattr(file, "name"):  # Older file objects expose the path in `.name`.
            paths.append(file.name)  # Store the temporary uploaded file path.
        elif isinstance(file, dict) and file.get("path"):  # Some versions return dictionaries.
            paths.append(file["path"])  # Store the path value from the dictionary.
    return paths  # Return normalized paths for local attachment loading.


def _toggle_visibility(enabled: bool):
    """Show integration-specific inputs only when their toggle is selected."""

    return gr.update(visible=enabled)  # Gradio uses this update object to show/hide groups.


def _toggle_prompt_source(prompt_source: str):
    """Switch visible prompt inputs between manual details and Jira ticket key."""

    use_manual = prompt_source == "Manual"  # Manual mode uses pasted issue details and logs.
    return gr.update(visible=use_manual), gr.update(visible=not use_manual)  # Return updates for manual and Jira groups.


def _section_text(sections: dict[str, str], names: list[str]) -> str:
    """Join selected parsed LLM sections into one textbox value."""

    values = []  # Keep non-empty requested sections in display order.
    for name in names:  # Walk each requested logical section.
        value = sections.get(name, "").strip()  # Read the parsed text for the section.
        if value:  # Skip missing sections so boxes stay clean.
            values.append(value)  # Add the section body to the combined output.
    return "\n\n".join(values).strip()  # Separate multiple sections with a readable blank line.


def _parse_analysis_sections(analysis: str) -> tuple[str, str, str, str]:
    """Split the LLM response into the four requested output textboxes."""

    headings = [  # These names mirror the prompt's requested output format.
        "Clarifications",
        "Summary",
        "Probable Root Causes",
        "Recommended Fix",
        "Code Lines To Inspect",
        "Verification Steps",
        "QA Guidelines",
    ]
    sections: dict[str, str] = {}  # Store heading-to-body mappings.
    escaped = "|".join(re.escape(heading) for heading in headings)  # Build safe regex choices.
    pattern = re.compile(  # Match numbered, markdown, or plain section headings.
        rf"(?im)^\s*(?:\d+\.\s*)?(?:#+\s*)?({escaped})\s*:?\s*$"
    )
    matches = list(pattern.finditer(analysis or ""))  # Locate all headings in the LLM text.

    if not matches:  # If the LLM returns unexpected formatting, keep the full answer visible.
        return "", "", analysis or "", ""

    for index, match in enumerate(matches):  # Slice each section body between adjacent headings.
        name = match.group(1)  # Capture the normalized heading name.
        start = match.end()  # Section text starts after the heading line.
        end = matches[index + 1].start() if index + 1 < len(matches) else len(analysis)  # Stop at next heading.
        sections[name] = analysis[start:end].strip()  # Save the cleaned body text.

    clarifications = _section_text(sections, ["Clarifications"])  # Textbox 1.
    summary = _section_text(sections, ["Summary"])  # Textbox 2.
    fix_details = _section_text(  # Textbox 3 combines root cause, fix, and code lines.
        sections,
        ["Probable Root Causes", "Recommended Fix", "Code Lines To Inspect"],
    )
    verification = _section_text(sections, ["Verification Steps", "QA Guidelines"])  # Textbox 4.
    return clarifications, summary, fix_details, verification  # Return values in UI output order.


def run_analysis(
    prompt_source,
    incident_desc,
    error_logs,
    jira_key,
    files,
    use_repository,
    repository_provider,
    repository,
    ref,
    file_path,
    start_line,
    end_line,
):
    try:
        use_jira = prompt_source == "Jira"  # Jira mode fetches ticket content for prompt construction.
        manual_desc = incident_desc if prompt_source == "Manual" else ""  # Avoid mixing manual text into Jira mode.
        manual_logs = error_logs if prompt_source == "Manual" else ""  # Avoid mixing manual logs into Jira mode.
        result = analyze_incident(
            incident_desc=manual_desc or "",
            error_logs=manual_logs or "",
            jira_key=(jira_key or "").strip(),
            local_attachment_paths=_file_paths(files),
            repository_provider=(repository_provider or "").strip(),
            repository=(repository or "").strip(),
            file_path=(file_path or "").strip(),
            ref=(ref or "main").strip(),
            start_line=int(start_line) if start_line else None,
            end_line=int(end_line) if end_line else None,
            include_jira=bool(use_jira),
            include_repository=bool(use_repository),
        )
        parsed_sections = _parse_analysis_sections(result.analysis)  # Split one LLM response into four boxes.
        cost_summary = _format_cost_summary(result.usage)  # Format token/cost details for the cost section.
        return (*parsed_sections, cost_summary)  # Return analysis boxes plus usage/cost textbox.
    except Exception as exc:
        error = f"Analysis failed: {exc}"  # Make failures visible in the first output box.
        return error, "", "", "", ""  # Keep the remaining output boxes empty on failure.


def _format_cost_summary(usage) -> str:
    """Format LiteLLM token and cost metadata for the Gradio cost section."""

    return "\n".join(  # Multiline text keeps the cost section easy to scan.
        [
            f"Model: {usage.model or 'Unknown'}",
            f"Input Tokens: {usage.input_tokens}",
            f"Output Tokens: {usage.output_tokens}",
            f"Total Tokens: {usage.total_tokens}",
            f"Estimated Cost USD: ${usage.estimated_cost_usd:.6f}",
        ]
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Incident Analyzer") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="hero">
                    <div class="eyebrow">Incident intelligence workspace</div>
                    <h1>AI Incident Analyzer</h1>
                    <p>Convert logs, screenshots, Jira context, and source snippets into practical fixes with visible token usage and cost.</p>
                </div>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=2, elem_classes=["panel", "input-card"]):
                    gr.Markdown("### Prompt Source")
                    prompt_source = gr.Radio(
                        label="Build Prompt From",
                        choices=["Manual", "Jira"],
                        value="Manual",
                    )
                    with gr.Group(visible=True) as manual_group:
                        incident_desc = gr.Textbox(
                            label="Issue Description",
                            lines=5,
                            placeholder="Describe the failure, user impact, environment, recent change, or observed behavior.",
                        )
                        error_logs = gr.Textbox(
                            label="Pasted Logs or Stack Trace",
                            lines=5,
                            placeholder="Paste the most relevant error logs, stack trace, failed job output, or exception details.",
                        )
                    with gr.Group(visible=False) as jira_group:
                        jira_key = gr.Textbox(
                            label="Jira Ticket Key",
                            placeholder="PROJ-123",
                        )

                with gr.Column(scale=1, elem_classes=["panel", "input-card"]):
                    gr.Markdown("### Evidence Uploads")
                    files = gr.File(
                        label="Upload Log Files or Screenshots",
                        file_count="multiple",
                        file_types=[
                            ".txt",
                            ".log",
                            ".json",
                            ".xml",
                            ".yaml",
                            ".yml",
                            ".csv",
                            ".md",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                        ],
                    )
                    gr.Markdown(
                        "Text files are added to the prompt. Images are sent as visual evidence when the selected LLM supports image input."
                    )

                with gr.Column(scale=1, elem_classes=["panel", "input-card", "repo-card"]):
                    use_repository = gr.Checkbox(label="Include GitHub/GitLab code context", value=False)
                    with gr.Group(visible=False) as repo_group:
                        repository_provider = gr.Dropdown(
                            label="Repository Provider",
                            choices=["github", "gitlab"],
                            value="github",
                        )
                        repository = gr.Textbox(
                            label="Repository Link or Path",
                            lines=1,
                            placeholder="owner/repo or group/project",
                        )
                        ref = gr.Textbox(label="Ref", value="main", lines=1)
                        file_path = gr.Textbox(label="File Path", lines=1, placeholder="src/module/file.py")
                        with gr.Row():
                            start_line = gr.Number(label="Start Line", precision=0)
                            end_line = gr.Number(label="End Line", precision=0)

            analyze_button = gr.Button("Submit Incident Analysis", variant="primary", elem_id="submit-btn")

            with gr.Column(elem_classes=["cost-card"]):
                gr.Markdown("### Cost and Token Usage")
                cost_output = gr.Textbox(
                    label="LiteLLM Usage Summary",
                    lines=5,
                    interactive=False,
                    elem_classes=["cost-box"],
                )

            gr.Markdown("### Analysis Output")
            with gr.Row(equal_height=True):
                clarifications_output = gr.Textbox(
                    label="Clarifications",
                    lines=5,
                    interactive=False,
                    elem_classes=["output-box"],
                )
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=5,
                    interactive=False,
                    elem_classes=["output-box"],
                )
            with gr.Row(equal_height=True):
                fix_output = gr.Textbox(
                    label="Probable Root Causes, Recommended Fix, Code Lines To Inspect",
                    lines=5,
                    interactive=False,
                    elem_classes=["output-box"],
                )
                verification_output = gr.Textbox(
                    label="Verification Steps and QA Guidelines",
                    lines=5,
                    interactive=False,
                    elem_classes=["output-box"],
                )

            prompt_source.change(_toggle_prompt_source, inputs=prompt_source, outputs=[manual_group, jira_group])
            use_repository.change(_toggle_visibility, inputs=use_repository, outputs=repo_group)
            analyze_button.click(
                run_analysis,
                inputs=[
                    prompt_source,
                    incident_desc,
                    error_logs,
                    jira_key,
                    files,
                    use_repository,
                    repository_provider,
                    repository,
                    ref,
                    file_path,
                    start_line,
                    end_line,
                ],
                outputs=[
                    clarifications_output,
                    summary_output,
                    fix_output,
                    verification_output,
                    cost_output,
                ],
            )

    return demo


def launch_demo():
    """Launch the UI with the configured theme and CSS."""

    return build_demo().launch(theme=APP_THEME, css=APP_CSS)


if __name__ == "__main__":
    launch_demo()
