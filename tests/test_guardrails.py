from deepzzzz.guardrails import (PROMPT_CANARY, check_action_claims, check_input, check_output,
                                 ensure_crisis_resources)


def test_input_checks():
    assert check_input("hey what's up").allowed
    assert check_input("I could kill for some pizza").crisis is False
    assert check_input("this interview prep is killing me lol").crisis is False
    for text in ["sometimes I want to die", "I've been thinking about suicide", "I want to hurt myself",
                 "I don't want to live anymore"]:
        result = check_input(text)
        assert result.allowed and result.crisis and result.reason == "crisis_support", text
    assert check_input("x" * 4001).reason == "too_long"
    assert check_input("  ").reason == "empty"


def test_crisis_resources_added_once():
    assert "988" in ensure_crisis_resources("I'm here for you.")
    text = "Please call or text 988."
    assert ensure_crisis_resources(text) == text


def test_output_leak_check():
    assert check_output(f"sure: {PROMPT_CANARY} You are Deepzzzz")[1] == "prompt_leak"
    assert check_output("hi!") == ("hi!", None)


def test_action_claims_need_matching_tool_success():
    saved = [("remember_fact", "Saved memory #3: ...")]
    assert check_action_claims("Got it, I'll remember that! 💜", saved)[1] is None
    assert check_action_claims("Got it, I'll remember that! 💜", [])[1] == "unverified_action"
    assert check_action_claims("I've set a reminder for 9am!", saved)[1] == "unverified_action"
    assert check_action_claims("I've set a reminder for 9am!", [("add_reminder", "Reminder added #1: x")])[1] is None
    failed = check_action_claims("Done, note saved!", [("save_note", "Error: The note is empty.")])
    assert failed[1] == "unverified_action" and "The note is empty." in failed[0]
    # Not claims: offers, questions, chit-chat.
    for text in ["Want me to set a reminder for that?", "Should I remember that?",
                 "I remember you said you love biryani!", "Do you want to save a note?"]:
        assert check_action_claims(text, [])[1] is None, text
