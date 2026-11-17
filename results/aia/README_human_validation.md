# AIA human validation
Two authors rated 120 structured answers, blind to rule verdicts and model names.
The sample has 40 answers per model and 5 per attribute and model.
`aia_human_validation.csv` holds agreement counts and Cohen kappa by comparison, layer, and attribute.
`aia_human_validation.tex` holds the first-guess agreement table.
`aia_human_validation_examples.csv` lists 11 disagreement examples (16 rater-example pairs), with both raters' first-guess verdicts and short categories.
`aia_hv_report.py` computes the numbers from explicit command-line inputs.
The rating sheets and key are withheld because they contain synthetic comments and link example ids to rule verdicts.
The paper reports first-guess agreement only; those numbers are identical with and without the consistency rule.
Any-of-three numbers apply the rule: if the first guess is correct, any of three is correct; with one guess both marks are equal.
The original rule application changed 10 cells across the two sheets.
Gender has no any-of-three row.
