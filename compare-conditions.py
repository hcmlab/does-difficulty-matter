import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns

import data_utils as d_utils

label_cols = ["nlg", "cnt_set", "cnt_unset", "cnt_reset", "flow_score", "ncs_score", "mental_score", "temporal_score", "performance_score", "effort_score", "frustration_score", "difficulty_score", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "age"]
label_names = ["Normalized Learning Gain",
			   "Number of sets",
			   "Number of unsets",
			   "Number of resets",
			   "Flow",
			   "Need for cognition",
			   "Perceived mental demand",
			   "Perceived temporal demand",
			   "Perceived performance",
			   "Perceived effort",
			   "Perceived frustration",
			   "Perceived difficulty",
			   "I have heard of graph theory before", 
			   "I like mathematics",
			   "I like puzzle games",
			   "I know how to procedurally solve puzzles (Sudoku, Rubik's cube, etc.)",
			   "I have experience in programming",
			   "I have played strategy games before (Chess, Go, Red Alert, etc.)",
			   "Age"]
label_ranges = [None for _ in label_cols]
label_ranges[label_cols.index("nlg")] = [-1, 1]
label_ranges[label_cols.index("flow_score")] = [0, 7]
label_ranges[label_cols.index("difficulty_score")] = [0, 20]

ratio_groups = []
sigdiff_pairs = [[] for _ in label_cols]

sigdiff_pairs[label_cols.index("nlg")] = [[0, 4], [1, 4], [3, 4]]

ans, questionnaire, thetas = d_utils.load_data(suffix = "unfiltered")

ans, ans_train, ans_test = d_utils.answer_preprocess(ans)
ans_train = d_utils.answer_augment_4pl(ans_train, thetas)
questionnaire = d_utils.questionnaire_preprocess(questionnaire)
ans_agg = d_utils.answer_aggregate(ans_train, ans_test, questionnaire)
ans_click_times = pd.read_csv("data/click_times.csv")

ans_agg = pd.merge(ans_agg, ans_click_times, how = "inner", on = ["user_id"])

# Remove rows with a nan
ans_agg = ans_agg.loc[~ans_agg[label_cols].isnull().any(axis = 1)]

for measure in ["nlg", "flow_score", "difficulty_score"]:
	measure_conditions = [ans_agg.loc[ans_agg["condition"] == condition, measure] for condition in ["predef", "self-det", "dda"]]

	stat, p = stats.kruskal(*measure_conditions)

	# Eta-squared effect size
	eta_sq = (stat - 3 + 1) / (len(ans_agg) - 3)

	print("%s Normality" % measure, stats.shapiro(ans_agg[measure]))
	print("%s Bartlett" % measure, stats.bartlett(*measure_conditions))
	print("%s ANOVA" % measure, stats.f_oneway(*measure_conditions))
	print(round(pg.anova(dv = measure, between = "condition", data = ans_agg, detailed = True, effsize='n2'), 4))
	print("%s Kruskal-Wallis" % measure, "H-stat = %.2f, p = %.3f, eta-squared = %.3f" % (stat, p, eta_sq))

# Compute ratio for each ratio group
for ratio_group in ratio_groups:
	cols_sum = np.sum(ans_agg[ratio_group].to_numpy(), axis = 1)
	ans_agg.loc[:, ratio_group] /= np.expand_dims(cols_sum, 1)

for label_col, label_name, label_range, sigdiff in zip(label_cols, label_names, label_ranges, sigdiff_pairs):
	# plt.figure(figsize = (5, 3))
	plt.clf()
	sns.set(rc = { "figure.figsize": (1.619, 1) })
	sns.barplot(data = ans_agg, x = "condition", y = label_col, palette = "colorblind")

	if label_range is not None:
		plt.ylim(label_range)

	plt.title("%s by condition" % label_name)
	plt.savefig("plots/conditions/%s.png" % label_col, bbox_inches = "tight", dpi = 300)

plt.figure(figsize = (5, 3))
sns.histplot(data = ans_agg, x = "condition", hue = "gender", multiple = "dodge", shrink = 0.8, palette = "colorblind")
plt.title("Gender by condition")
plt.savefig("plots/conditions/gender.png", bbox_inches = "tight", dpi = 300)

for condition in ["predef", "self-det", "dda"]:
	print("%s - Male %d, female %d" % (condition, sum((ans_agg["condition"] == condition) & (ans_agg["gender"] == "m")), sum((ans_agg["condition"] == condition) & (ans_agg["gender"] == "f"))))

print("Age average %f" % ans_agg["age"].mean())