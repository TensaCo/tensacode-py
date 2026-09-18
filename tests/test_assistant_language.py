"""The chat assistant's forgiving grammar: utterance -> request frames.

Each case lists the expected frames as (act, slots-subset). Unclear requests must come
back ``unknown`` rather than as a guessed action, and never as a destructive one.
"""

import pytest

from examples.browser_agents.assistant.language import parse_message

CASES = [
    # --- create folders
    ("make a folder called recipes on my desktop", [("create_folder", {"name": "recipes", "place": "~/Desktop"})]),
    ('create a new folder named "Tax 2026" in documents', [("create_folder", {"name": "Tax 2026", "place": "~/Documents"})]),
    ("new folder photos", [("create_folder", {"name": "photos"})]),
    ("please make a directory called scratch in ~/Projects", [("create_folder", {"name": "scratch", "place": "~/Projects"})]),
    ("could you create a folder named backups in my home folder?", [("create_folder", {"name": "backups", "place": "~"})]),
    ("mkdir foo", [("create_folder", {"name": "foo"})]),
    ("make a folder called a and a folder called b", [("create_folder", {"name": "a"}), ("create_folder", {"name": "b"})]),
    ("create folders x, y and z", [("create_folder", {"name": "x"}), ("create_folder", {"name": "y"}), ("create_folder", {"name": "z"})]),
    ("make folders called drafts and final on the desktop", [("create_folder", {"name": "drafts", "place": "~/Desktop"}), ("create_folder", {"name": "final", "place": "~/Desktop"})]),
    ("make a fodler called music2 on my desktp", [("create_folder", {"name": "music2", "place": "~/Desktop"})]),
    ("craete a folder called invoices in documnets", [("create_folder", {"name": "invoices", "place": "~/Documents"})]),
    ("make a folder called 'old stuff' inside the recipes folder", [("create_folder", {"name": "old stuff", "place": "@recipes"})]),
    ("add a folder called logs in it", [("create_folder", {"name": "logs", "place": "@it"})]),
    # --- create files
    ('can you make a file called shopping.txt in it with the text "eggs, milk"', [("create_file", {"name": "shopping.txt", "place": "@it", "text": "eggs, milk"})]),
    ("make a file notes.txt on the desktop containing hello world", [("create_file", {"name": "notes.txt", "place": "~/Desktop", "text": "hello world"})]),
    ("please create todo.md in ~/Projects/demo", [("create_file", {"name": "todo.md", "place": "~/Projects/demo"})]),
    ("touch a.txt", [("create_file", {"name": "a.txt"})]),
    ("create an empty file called log.txt", [("create_file", {"name": "log.txt"})]),
    ("put hello in a file called hi.txt", [("create_file", {"name": "hi.txt", "text": "hello"})]),
    ("put 'don't forget the milk' in a new file called reminder.txt on the desktop", [("create_file", {"name": "reminder.txt", "text": "don't forget the milk", "place": "~/Desktop"})]),
    ("make a readme in it", [("create_file", {"name": "README.md", "place": "@it"})]),
    ('create a file named "meeting notes.txt" saying "call Bob"', [("create_file", {"name": "meeting notes.txt", "text": "call Bob"})]),
    ("new file ideas.md", [("create_file", {"name": "ideas.md"})]),
    # --- write / append
    ('add "buy bread" to shopping.txt', [("write", {"text": "buy bread", "target": "shopping.txt", "append": True})]),
    ('write "hello" to ~/Desktop/hi.txt', [("write", {"text": "hello", "target": "~/Desktop/hi.txt", "append": False})]),
    ("append 'line two' to it", [("write", {"text": "line two", "target": "@it", "append": True})]),
    ("add \"don't forget\" to the end of notes.txt", [("write", {"text": "don't forget", "target": "notes.txt", "append": True})]),
    ('write "a b c" into "my notes.txt"', [("write", {"text": "a b c", "target": "my notes.txt", "append": False})]),
    # --- list
    ("what's on my desktop?", [("list", {"place": "~/Desktop"})]),
    ("whats on the desktop", [("list", {"place": "~/Desktop"})]),
    ("list the files in documents", [("list", {"place": "~/Documents"})]),
    ("show me what's in the recipes folder", [("list", {"place": "@recipes"})]),
    ("how many files are in downloads", [("list", {"place": "~/Downloads", "count": True})]),
    ("whats in it", [("list", {"place": "@it"})]),
    ("show it", [("list", {"place": "@it"})]),
    ("ls ~/Documents", [("list", {"place": "~/Documents"})]),
    ("ls", [("list", {})]),
    ("open the documents folder", [("list", {"place": "~/Documents"})]),
    ("what is in my downloads folder", [("list", {"place": "~/Downloads"})]),
    ("list ~/Projects", [("list", {"place": "~/Projects"})]),
    ("show the contents of the desktop", [("list", {"place": "~/Desktop"})]),
    ("what's in ~/Documents/old", [("list", {"place": "~/Documents/old"})]),
    # --- read
    ("read shopping.txt", [("read", {"target": "shopping.txt"})]),
    ("what does ~/Documents/trajectory-task.txt say", [("read", {"target": "~/Documents/trajectory-task.txt"})]),
    ("open notes.txt", [("read", {"target": "notes.txt"})]),
    ("cat notes.txt", [("read", {"target": "notes.txt"})]),
    ("read it", [("read", {"target": "@it"})]),
    ("show me notes.txt on the desktop", [("read", {"target": "notes.txt", "place": "~/Desktop"})]),
    ('open "meeting notes.txt"', [("read", {"target": "meeting notes.txt"})]),
    ("print the contents of the file todo.md", [("read", {"target": "todo.md"})]),
    # --- delete
    ("delete the recipes folder", [("delete", {"target": "recipes"})]),
    ("remove shopping.txt from the desktop", [("delete", {"target": "shopping.txt"})]),
    ("delete it", [("delete", {"target": "@it"})]),
    ("delete everything on my desktop", [("delete", {"target": "~/Desktop"})]),
    ("rm foo.txt", [("delete", {"target": "foo.txt"})]),
    ("rm -r old", [("delete", {"target": "old"})]),
    ("get rid of ~/Desktop/tmp", [("delete", {"target": "~/Desktop/tmp"})]),
    ("please delte notes.txt", [("delete", {"target": "notes.txt"})]),
    ("trash that file", [("delete", {"target": "@it"})]),
    # --- move / copy / rename
    ("rename notes.txt to ideas.txt", [("rename", {"target": "notes.txt", "new_name": "ideas.txt"})]),
    ("rename it to final.txt", [("rename", {"target": "@it", "new_name": "final.txt"})]),
    ("move ideas.txt to documents", [("move", {"target": "ideas.txt", "dest": "~/Documents"})]),
    ("copy ~/Documents/research-note.pdf to the desktop", [("copy", {"target": "~/Documents/research-note.pdf", "dest": "~/Desktop"})]),
    ("move it into the recipes folder", [("move", {"target": "@it", "dest": "@recipes"})]),
    ("duplicate report.pdf to ~/Downloads", [("copy", {"target": "report.pdf", "dest": "~/Downloads"})]),
    ("mv a.txt b.txt", [("rename", {"target": "a.txt", "new_name": "b.txt"})]),
    ("cp a.txt ~/Documents", [("copy", {"target": "a.txt", "dest": "~/Documents"})]),
    # --- find / grep
    ("find all pdfs", [("find", {"pattern": "*.pdf", "place": "~"})]),
    ("where is trajectory-task.txt", [("find", {"pattern": "trajectory-task.txt", "place": "~"})]),
    ("find files named report.pdf in documents", [("find", {"pattern": "report.pdf", "place": "~/Documents"})]),
    ("locate todo.md", [("find", {"pattern": "todo.md", "place": "~"})]),
    ('search for "dns" in documents', [("grep", {"needle": "dns", "place": "~/Documents"})]),
    ("which files mention dashboard", [("grep", {"needle": "dashboard", "place": "~"})]),
    ('look for "budget 2026" on my desktop', [("grep", {"needle": "budget 2026", "place": "~/Desktop"})]),
    # --- count / size
    ("how big is the documents folder", [("size", {"target": "~/Documents"})]),
    ("how many lines are in ~/Documents/trajectory-task.txt", [("count", {"unit": "lines", "target": "~/Documents/trajectory-task.txt"})]),
    ("count the words in notes.txt", [("count", {"unit": "words", "target": "notes.txt"})]),
    ("what's the size of it", [("size", {"target": "@it"})]),
    ("how large is report.pdf", [("size", {"target": "report.pdf"})]),
    # --- info
    ("how much disk space do I have", [("info", {"topic": "disk"})]),
    ("what time is it", [("info", {"topic": "date"})]),
    ("what's the date today?", [("info", {"topic": "date"})]),
    ("who am i", [("info", {"topic": "user"})]),
    ("what's my ip address", [("info", {"topic": "ip"})]),
    ("what's running", [("info", {"topic": "processes"})]),
    ("how many cpu cores are there", [("info", {"topic": "cpus"})]),
    ("what's the hostname", [("info", {"topic": "hostname"})]),
    ("where am i", [("info", {"topic": "cwd"})]),
    ("pwd", [("info", {"topic": "cwd"})]),
    ("what operating system is this", [("info", {"topic": "os"})]),
    ("uptime", [("info", {"topic": "uptime"})]),
    ("is git installed", [("which", {"program": "git"})]),
    # --- apps
    ("open firefox", [("open_app", {"app": "Firefox"})]),
    ("launch the text editor", [("open_app", {"app": "Text Editor"})]),
    ("start vs code", [("open_app", {"app": "Visual Studio Code"})]),
    ("open the file manager", [("open_app", {"app": "Files"})]),
    # --- git
    ("make recipes a git repo", [("git_init", {"target": "@recipes"})]),
    ('commit everything in recipes with message "first"', [("git_commit", {"target": "@recipes", "message": "first"})]),
    ("what changed in recipes", [("git_status", {"target": "@recipes"})]),
    ("show the commit history of recipes", [("git_log", {"target": "@recipes"})]),
    ("git init in ~/Projects/app", [("git_init", {"target": "~/Projects/app"})]),
    ("commit it as 'wip'", [("git_commit", {"target": "@it", "message": "wip"})]),
    ("initialize a git repo there", [("git_init", {"target": "@it"})]),
    # --- cd
    ("go to documents", [("cd", {"target": "~/Documents"})]),
    ("cd ~/Projects", [("cd", {"target": "~/Projects"})]),
    ("go home", [("cd", {"target": "~"})]),
    ("go back", [("cd", {"target": "~"})]),
    ("switch to the recipes folder", [("cd", {"target": "@recipes"})]),
    # --- run / install
    ("run `ls -la ~`", [("run", {"command": "ls -la ~"})]),
    ("`git status`", [("run", {"command": "git status"})]),
    ("$ echo hi", [("run", {"command": "echo hi"})]),
    ("sudo apt update", [("run", {"command": "sudo apt update"})]),
    ("install cowsay", [("install", {"package": "cowsay"})]),
    # --- multi-clause
    ("make a folder called projects2 then put a file called a.txt in it and then show me what's in it",
     [("create_folder", {"name": "projects2"}), ("create_file", {"name": "a.txt", "place": "@it"}), ("list", {"place": "@it"})]),
    ("create notes.txt on the desktop and add 'hello' to it", [("create_file", {"name": "notes.txt", "place": "~/Desktop"}), ("write", {"text": "hello", "target": "@it", "append": True})]),
    ("go to documents. list the files", [("cd", {"target": "~/Documents"}), ("list", {})]),
    ("make a folder called site, then make it a git repo", [("create_folder", {"name": "site"}), ("git_init", {"target": "@it"})]),
    # --- project notes stay whole
    ("set up a project called ledger-lab under ~/Projects. README.md should say 'Ledger Lab'. todo: a; b. commit with message 'init'", [("setup_project", {})]),
    # --- dialogue
    ("hi", [("greet", {})]),
    ("hello there!", [("greet", {})]),
    ("thanks!", [("thanks", {})]),
    ("thank you", [("thanks", {})]),
    ("yes", [("confirm", {})]),
    ("yes please", [("confirm", {})]),
    ("go ahead", [("confirm", {})]),
    ("no", [("cancel", {})]),
    ("never mind", [("cancel", {})]),
    ("what can you do", [("help", {})]),
    ("help", [("help", {})]),
    ("1", [("choose", {})]),
    ("the second one", [("choose", {})]),
    ("2nd", [("choose", {})]),
    ("the last one", [("choose", {})]),
    ("the one on the desktop", [("choose", {})]),
    ("~/Desktop/notes.txt", [("read", {"target": "~/Desktop/notes.txt"})]),  # the agent treats it as the answer when a "which one?" is open
    ("number 3", [("choose", {})]),
    ("what's in the folder called 'my files'", [("list", {"place": "@my files"})]),
    ("delete the file", [("delete", {"target": "@it"})]),
    ("yes, delete it", [("confirm", {})]),
    ("no, leave it", [("cancel", {})]),
    ("remove all files on my desktop", [("delete", {"target": "~/Desktop"})]),
    ("make a folder called desktp", [("create_folder", {"name": "desktp"})]),
    # --- honest unknowns (never a guessed action)
    ("make me a sandwich", [("unknown", {})]),
    # these two were unknown until a clarification faculty existed: with one, the honest reading is
    # "understood but underdetermined", which asks how rather than guessing (see docs/revival/24)
    ("organize my desktop", [("clarify_goal", {"place": "~/Desktop"})]),
    ("what's the weather", [("unknown", {})]),
    ("clean up my desktop", [("clarify_goal", {"place": "~/Desktop"})]),
    ("remove the background from photo.png", [("unknown", {})]),
    ("move on", [("unknown", {})]),
    ("run the tests", [("unknown", {})]),
    ("copy the time sheet", [("unknown", {})]),
    ("tell me a joke", [("unknown", {})]),
    ("fix my wifi", [("unknown", {})]),
]


def _ids():
    return [c[0][:50] for c in CASES]


@pytest.mark.parametrize("text,expected", CASES, ids=_ids())
def test_parse(text, expected):
    frames = parse_message(text)
    got = [(f.act, f.slots) for f in frames]
    assert len(frames) == len(expected), got
    for frame, (act, slots) in zip(frames, expected):
        assert frame.act == act, got
        for k, v in slots.items():
            assert frame.slots.get(k) == v, (k, got)


DESTRUCTIVE = {"delete", "move", "rename", "run", "install", "write"}


@pytest.mark.parametrize("text", ["organize my desktop", "clean up my desktop", "remove the background from photo.png", "move on", "run the tests", "tidy things up", "get rid of the noise", "move fast and break things"])
def test_unclear_requests_are_never_destructive(text):
    assert not {f.act for f in parse_message(text)} & DESTRUCTIVE


def test_choose_keeps_the_answer_words():
    (frame,) = parse_message("the second one")
    assert frame.act == "choose" and frame.words == "the second one"
