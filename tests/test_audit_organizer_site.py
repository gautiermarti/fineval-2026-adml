import json

import pytest

from scripts.audit_organizer_site import effective_template, parse_effective_site


def bundled(template: str, override: str = "") -> str:
    return (
        "<html><script>" + override + "</script>"
        '<script type="__bundler/template">'
        + json.dumps(template)
        + "</script></html>"
    )


def minimal_template(date: str = "04 AUG 2026", submission: str = "TBD") -> str:
    solution = (
        f'<a href="{submission}">Submit</a>'
        if submission.startswith("http") else f"<span>{submission}</span>"
    )
    return f'''
      <section id="tasks">
        link: "https://drive.google.com/drive/folders/task1?usp=sharing"
        link: "https://drive.google.com/drive/folders/task2?usp=sharing"
        link: "https://drive.google.com/drive/folders/task3?usp=sharing"
      </section>
      <div id="dates">
        {{ date: "{date}", event: "Test set release, test phase starts", status: "PENDING" }}
        {{ date: "23 AUG 2026", event: "Test phase ends", status: "PENDING" }}
      </div>
      <section id="submission">Solution submission {solution}
        System paper submission
      </section>
      <section id="leaderboard"></section>
    '''


def test_applies_outer_string_override_before_parsing() -> None:
    old = '{ date: "04 AUG 2026", event: "Test set release, test phase starts", status: "PENDING" }'
    new = '{ date: "08 AUG 2026", event: "Test set release, test phase starts", status: "PENDING" }'
    override = f"replaceTemplateSection('{old}', '{new}', 'test set release date');"
    result = parse_effective_site(bundled(minimal_template(), override))
    assert result["event_dates"]["Test set release, test phase starts"] == "08 AUG 2026"
    assert result["submission_status"] == "TBD"
    assert result["all_parsed_overrides_found"] is True


def test_extracts_available_submission_link() -> None:
    url = "https://example.test/submit"
    result = parse_effective_site(bundled(minimal_template(submission=url)))
    assert result["submission_status"] == "available"
    assert result["submission_url"] == url
    assert result["submission_urls"] == [url]
    assert result["drive_folder_urls"] == [
        "https://drive.google.com/drive/folders/task1",
        "https://drive.google.com/drive/folders/task2",
        "https://drive.google.com/drive/folders/task3",
    ]


def test_extracts_task_space_submission_links() -> None:
    urls = [
        "https://huggingface.co/spaces/org/task1",
        "https://huggingface.co/spaces/org/task2",
        "https://huggingface.co/spaces/org/task3",
    ]
    template = minimal_template()
    links = "".join(f'<a href="{url}">Task</a>' for url in urls)
    template = template.replace(
        "System paper submission", f"Test set submission {links} System paper submission"
    )
    result = parse_effective_site(bundled(template))
    assert result["submission_status"] == "available"
    assert result["submission_url"] == urls[0]
    assert result["task_submission_urls"] == urls


def test_applies_template_literal_override() -> None:
    urls = [
        "https://huggingface.co/spaces/org/task1",
        "https://huggingface.co/spaces/org/task2",
        "https://huggingface.co/spaces/org/task3",
    ]
    before = "System paper submission"
    links = "".join(f'<a href="{url}">Task</a>' for url in urls)
    after = f"Test set submission {links} System paper submission"
    override = f"replaceTemplateSection(`{before}`, `{after}`, `submission links`);"
    result = parse_effective_site(bundled(minimal_template(), override))
    assert result["task_submission_urls"] == urls
    assert result["all_parsed_overrides_found"] is True


def test_missing_bundle_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="template"):
        effective_template("<html></html>")


def test_drive_folder_count_is_fail_closed() -> None:
    template = minimal_template().replace(
        'link: "https://drive.google.com/drive/folders/task3?usp=sharing"', ""
    )
    with pytest.raises(ValueError, match="three organizer Drive folders"):
        parse_effective_site(bundled(template))
