# Text Input

`alice_120.txt` contains 120 sentence-like segments extracted from the
public-domain Project Gutenberg text of Lewis Carroll's *Alice's Adventures in
Wonderland*:

https://www.gutenberg.org/ebooks/11

The experiment treats each non-empty line as one input sequence. The extraction
uses a simple rule-based sentence splitter, so dialogue fragments such as
`Oh dear!` may appear as separate lines. The file is fixed in the repository so
the paper defaults do not depend on network access.
