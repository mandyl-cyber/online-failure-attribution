"""AI annotations for changed claims in the frozen sidecar dev comparison.

Keys are zero-based pair/claim indices. The pipeline pins the reviewed claim
text in its review packet and reuses prior judgments only on exact matches.
"""

# support, note, severe. These are reviewer judgments, not model predictions.
JUDGMENTS = {
    (0, 1): ('partial', 'Reading is in_progress, but testing is todo in S2.', True),
    (6, 2): ('supported', 'S23 explicitly contains the named Galaxy test match.', False),
    (7, 1): ('partial', 'S25 explicitly says only 115 lines exist. Requested 1-280 is not observed output length.', False),
    (11, 0): ('supported', 'Cited source defines VariableManager.', False),
    (13, 1): ('partial', 'Cited action supports requested range 1-240, not the assertion that this output was printed.', False),
    (15, 0): ('supported', 'The Makefile source and command exit0 are present.', False),
    (17, 0): ('supported', 'Console edit receipts show both added expressions.', False),
    (17, 1): ('partial', 'Cited actions propose the edits. Receipts are in uncited observations.', False),
    (18, 0): ('supported', 'S37 edit succeeds. S38 command is missing. S39 replacement fails.', False),
    (18, 1): ('supported', 'Explicitly attributes these changes to the commit message.', False),
    (19, 0): ('partial', 'Plan testing is in_progress. Missing ansible comes from S42 observation, which is not cited here.', False),
    (19, 1): ('supported', 'S42 observation explicitly reports the missing ansible module.', False),
    (20, 0): ('supported', 'Read as the combined probe aborting on yaml import, with no independent tests completed.', False),
    (20, 1): ('unsupported', 'S44 marks all eight phases done, including phase8.', True),
    (21, 0): ('supported', 'S15 marks the cache reproduction phase todo.', False),
    (21, 1): ('partial', 'Directory listings exist. Expected has no demonstrated specification. The cited pytest error is omitted.', False),
    (27, 0): ('supported', 'Local/global sync/async, batch and abatch cases appear in source.', False),
    (27, 2): ('supported', 'Cited metadata reports command exit0.', False),
    (28, 0): ('partial', 'Action proposes the stated change. Completion is established by the separate receipt, not this citation.', False),
    (28, 1): ('supported', 'Receipt shows llm_cache definition and aget_prompts receiving it.', False),
    (30, 1): ('partial', 'Locations and source display exist, but first command was grep, not sed, and location sequence is misassigned.', False),
    (32, 0): ('supported', 'Helper exclusion refers to serialized state in __getstate__. text.py source is displayed.', False),
    (33, 0): ('partial', 'Receipt edits text.py serialization dictionary. It does not remove or set live leg._draggable.', True),
    (33, 1): ('partial', 'True and lengths are printed. Figure/annotation examples and method name are confused.', False),
    (37, 0): ('partial', 'S28 replacement fails. False is in uncited S29. S30 prints done, not false.', True),
    (37, 1): ('partial', 'S31 actually has phase2 in_progress, phases5/6 done, phase7 todo and phase8 todo.', True),
}
