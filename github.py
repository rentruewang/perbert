#! /usr/bin/env python

from datetime import datetime

from git import Repo

message = (
    f"🤖 RenTrueWang commits on {datetime.now()}"
    "\n\n"
    "Beep Boop, I'm a bot. This commit is created by me because @rentruewang is too lazy to write a commit message. Message over!"
)

repo = Repo(".")

repo.git.add(".")

repo.index.commit(message)

repo.remotes.origin.push()
