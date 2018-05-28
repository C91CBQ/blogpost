#!/usr/bin/python
import sys
import markdown2
fo = open(sys.argv[1], "r+")
str = fo.read()
html = markdown2.markdown(str, extras=["break-on-newline", "code-friendly", "fenced-code-blocks", "cuddled-lists", "target-blank-links"])
fo = open("output.html", "wb")
fo.write(html)
