"""My verdicts on the items that defeat every tier, so label noise is not counted as a gap.

The paraphrase set was written by a local model rewriting templates, with the template's label
carried over. On easy items that is fine. On the hard tail it is not: the rewrite often no
longer means what the template meant, and the carried label is then wrong. Grading a tier
against a wrong label manufactures a "schema gap" that does not exist.

Each entry below is one all-tiers-fail item, with my judgement and the reason. `gold_wrong`
means the recorded label does not describe the sentence; those items are excluded from the
measurement and counted as our own wiring error, in the class (d) sense of
docs/revival/11-evidence-audit.md.
"""

from __future__ import annotations

#: text -> (verdict, reason). verdict: gold_ok | gold_wrong | ambiguous
VERDICTS: dict[str, tuple[str, str]] = {
    # ---- the user's own prompts: gold is the user's report, and every one is a real gap
    "what do you know": ("gold_ok", "a question about the contents of its own memory; no topic vocabulary covers 'everything'"),
    "do you know who i am": ("gold_ok", "asks memory about the speaker, not the machine's logged-in user"),
    "am i Jacob": ("gold_ok", "a polar check against a remembered value, which no act expresses"),
    "my name isn't Jacob anymore": ("gold_ok", "a negated self-disclosure, which is a retraction rather than an assertion"),
    "what font is the terminal using": ("gold_ok", "asks a property of a screen region; only the region is representable, not the property"),
    "how big is the Files window": ("gold_ok", "'big' of a window is a screen measurement; all three tiers read it as file size"),
    "my name and my favourite colour": ("gold_ok", "two topics in one question; a slot holds one"),

    # ---- paraphrase items where the rewrite no longer means the template's label
    "Please carry out the task.": ("gold_wrong", "labelled confirm; the sentence is an instruction with no antecedent, not a yes"),
    "quick question, what's your take? thanks": ("gold_wrong", "labelled ask_screen; it asks for an opinion, which is not a screen question"),
    "Can you do it again?": ("gold_wrong", "labelled choose; it is a repeat request"),
    "Hey, can I get that now?": ("gold_wrong", "labelled confirm; it is a request for a previously discussed thing"),
    "You're welcome, thanks.": ("gold_wrong", "labelled confirm; the learned tier's `thanks` is the better reading"),
    "Hey, thanks!": ("gold_wrong", "labelled confirm; plainly thanks"),
    "could you please check that file?": ("gold_wrong", "labelled cd; it asks to inspect a file"),
    "Could you do it twice?": ("gold_wrong", "labelled choose; a repeat request"),
    "I was just wondering, do you remember my favourite colour now?": ("gold_wrong", "labelled forget; it asks whether the value is remembered"),
    "Could you turn the photos into a Git repository for me, please?": ("ambiguous", "gold `photos` vs regex `@photos`: the reference-kind convention, which is the gap itself"),
    "What's in the archive that's uncommitted?": ("ambiguous", "same reference-kind convention disagreement"),
    "I'd like you to make a folder called notes.": ("gold_wrong", "gold asserts place=@it though the sentence names no place; the template's context leaked in"),
    "Hi, could you create a file named todo.md for me?": ("gold_wrong", "same leaked place=@it"),
    "Create a file named README.md and write \"don't forget the milk\" inside it.": ("ambiguous", "gold place=@it is leaked, but the text slot is genuinely missed by two tiers"),
    "Can you please create a file named todo.md and put 'don't forget the milk' inside it?": ("ambiguous", "same shape: leaked place, genuine missed text"),
    "Please create a file named invoice.pdf containing the text “don't forget the milk”": ("gold_ok", "the content span is genuinely mis-bounded by two tiers ('the text ...')"),
    "Please write the todo: sleep into a new file named a.txt, thanks.": ("gold_ok", "content span mis-bounded; 'the todo: sleep' keeps its introducer"),
    "Could you run `ls ~/Desktop/recipes` for me?": ("ambiguous", "gold reads the backticked command as a list request; treating it as `run` is defensible"),
    "could you count the files in my temp folder please?": ("gold_ok", "place word 'temp' should canonicalise to /tmp; two tiers leave it a bare word"),
    "Hey, can you count the files in the photos folder?": ("gold_ok", "'photos' should canonicalise to ~/Pictures"),
    "quick question: what's the purpose of the sidebar?": ("ambiguous", "gold ask_pixels; 'purpose' is arguably not a pixel question"),
    "just curious, what's the purpose of the top bar for me?": ("ambiguous", "same"),
    "Please begin sending mail.": ("gold_ok", "an app named by its function; the app slot should be 'mail', not 'sending mail'"),
    "My email address is maya@seed.local.": ("gold_ok", "topic should canonicalise to 'email'; two tiers keep 'email address'"),
    "Hi, could you please share my email address?": ("gold_ok", "asks memory; the learned tier read it as forget, which is a wrong action on memory"),
    "please don't remember my name.": ("gold_ok", "a negated memory instruction: forget, not recall"),
    "quick query: are there any files with TODO comments? thanks.": ("gold_ok", "the needle should be TODO; 'TODO comments' over-captures"),
    # ---- second pass: the remaining all-tiers-fail items
    "Create a file named chapter1.md and write inside it the phrase \"salt the water\".": ("gold_wrong", "gold leaks place=@it; name and text are both read correctly by the learned tier"),
    "Put the 初期メモ into a new file named notes.txt.": ("gold_ok", "content span is non-Latin; the learned tier returns 'the', a real span-boundary failure"),
    "I'd like you to create a file named a.txt in my home directory.": ("ambiguous", "gold 'home' vs learned '~': the place-canonicalisation convention, which is the gap itself"),
    "Please create a file named report.pdf in the docs folder with the content \"don't forget the milk\".": ("gold_ok", "place 'docs' should canonicalise to ~/Documents; the learned tier drops place entirely"),
    "Could you copy it to ~/Projects, please?": ("gold_wrong", "labelled move; the sentence says copy"),
    "Move the file to ~/Desktop/recipes.": ("gold_wrong", "labelled copy; the sentence says move"),
    "Can you just know, Maya is my name now?": ("gold_ok", "a self-disclosure in an odd frame; read as a memory question instead of an assertion"),
    "Please create a folder named 'meeting notes.txt' for me.": ("gold_wrong", "gold leaks place=@it"),
    "I'd like you to copy it to ~": ("gold_wrong", "labelled move; the sentence says copy"),
    "Could you create a folder named \"desktopin photos\" on the desktop?": ("gold_wrong", "the paraphrase itself is corrupted ('desktopin'); no label can be right"),
    "Could you create a folder named “band practice” for me?": ("gold_wrong", "gold leaks place=@it"),
    "could you count the files in the music folder?": ("gold_ok", "'music' should canonicalise to ~/Music"),
    "create a file named notes.txt and write \"eggs, milk, bread\" inside it": ("gold_wrong", "gold leaks place=@it; name and text are read correctly"),
    "what's your take on it?": ("gold_wrong", "labelled ask_screen; it asks for an opinion"),
    "Could you open a terminal window?": ("ambiguous", "gold app='a terminal' keeps an article; the label itself is sloppy"),
    "Please create a folder named \"backupson\" in the music directory for me.": ("gold_wrong", "the paraphrase is corrupted ('backupson')"),
    "could you open my mail for me?": ("gold_ok", "an app named by possessive; the learned tier returns no app at all"),
}


def verdict(text: str) -> tuple[str, str]:
    return VERDICTS.get(text.strip(), ("unjudged", ""))


def counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for v, _ in VERDICTS.values():
        out[v] = out.get(v, 0) + 1
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(counts(), indent=1))
