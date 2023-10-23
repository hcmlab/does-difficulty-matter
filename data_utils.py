import itertools
import json
import numpy as np
import pandas as pd

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Load data from csv
def load_data(suffix = ""):
	if suffix != "":
		suffix = "_" + suffix

	with open("data/answer%s.csv" % suffix, "r") as fp:
		ans = pd.read_csv(fp)

	with open("data/questionnaire%s.csv" % suffix, "r") as fp:
		questionnaire = pd.read_csv(fp)

	with open("data/4pl_thetas%s.json" % suffix, "r") as fp:
		thetas = json.load(fp)

	return ans, questionnaire, thetas

# Preprocess answer data
# Additional columns and value renaming
# Calculate NLG
# Split between phases
def answer_preprocess(ans):
	ans.loc[ans["condition"] == "static-inc", "condition"] = "predef"

	ans["puzzle_cnt"] -= 3
	ans["extra_clicks"] = ans["clicks"] - ans["mis_size"]
	ans["max_is_ratio"] = ans["max_is"] / ans["mis_size"]
	ans["is_timeout"] = (ans["success"] == False) & (ans["time"] > 85)
	ans["is_incorrect"] = (ans["success"] == False) & (ans["time"] < 85)

	ans_train = ans[ans["phase"] == "train"]
	ans_test = ans[ans["phase"] != "train"]
	ans_test["lvl"] = ((ans_test["chosen_graph"] - 301) % 6) // 2

	# Calculate NLG
	ans_train_scores = ans_train.groupby("id_int")[["user_id", "condition", "success"]].agg({ "user_id": "min", "condition": "min", "success": "sum" }).rename(columns = { "success": "train"} )
	ans_train_scores.reset_index(inplace = True)

	ans_test_sort = ans_test.sort_values(by = ["id_int", "phase", "lvl"], ascending = [True, False, True])

	ans_test_sort_pretest = ans_test_sort.loc[ans_test_sort["phase"] == "pre-test"]
	ans_test_sort_posttest = ans_test_sort.loc[ans_test_sort["phase"] == "post-test"]

	# Label by difficulty
	for lvl, diff_name in zip([0, 1, 2], ["easy", "med", "hard"]):
		ans_test_sort_pretest["pre_%s" % diff_name] = 0
		ans_test_sort_pretest.loc[ans_test_sort_pretest["lvl"] == lvl, "pre_%s" % diff_name] = ans_test_sort_pretest.loc[ans_test_sort_pretest["lvl"] == lvl, "success"]

		ans_test_sort_posttest["post_%s" % diff_name] = 0
		ans_test_sort_posttest.loc[ans_test_sort_posttest["lvl"] == lvl, "post_%s" % diff_name] = ans_test_sort_posttest.loc[ans_test_sort_posttest["lvl"] == lvl, "success"]

	ans_test_pretest = ans_test_sort_pretest.groupby("id_int")[["condition", "success", "pre_easy", "pre_med", "pre_hard"]].agg({ "condition": "min", "success": "sum", "pre_easy": "sum", "pre_med": "sum", "pre_hard": "sum" }).rename(columns = { "success": "pre" }).reset_index()
	ans_test_posttest = ans_test_sort_posttest.groupby("id_int")[["user_id", "success", "post_easy", "post_med", "post_hard"]].agg({ "user_id": "min", "success": "sum", "post_easy": "sum", "post_med": "sum", "post_hard": "sum" }).rename(columns = { "success": "post" }).reset_index()

	ans_test_merge = ans_test_pretest.join(ans_test_posttest.set_index("id_int"), on = "id_int")
	ans_test_merge = ans_test_merge.join(ans_train_scores[["id_int", "train"]].set_index("id_int"), on = "id_int")
	
	ans_test_merge["nlg"] = 0.
	ans_test_merge["test_diff"] = ans_test_merge["post"] - ans_test_merge["pre"]
	ans_test_merge.loc[ans_test_merge["post"] > ans_test_merge["pre"], "nlg"] = ans_test_merge["test_diff"] / (3. - ans_test_merge["pre"])
	ans_test_merge.loc[ans_test_merge["post"] < ans_test_merge["pre"], "nlg"] = ans_test_merge["test_diff"] / (ans_test_merge["pre"])

	ans_test_merge = ans_test_merge.loc[(ans_test_merge["nlg"] != -1)]					# Filter out people with NLG -1 (2-sigma)
	ans_test_merge.set_index("id_int", inplace = True)

	ans_test = ans_test.merge(ans_test_merge[["pre", "post", "nlg", "test_diff", "train", "pre_easy", "pre_med", "pre_hard", "post_easy", "post_med", "post_hard"]], on = "id_int")

	return ans, ans_train, ans_test

# Preprocess questionnaire data
# Additional columns and value renaming
def questionnaire_preprocess(questionnaire):
	questionnaire.loc[questionnaire["condition"] == "static-inc", "condition"] = "predef"

	questionnaire["game_exp"] = questionnaire["exp3"] + questionnaire["exp4"] + questionnaire["exp6"]
	questionnaire["cs_exp"] = questionnaire["exp1"] + questionnaire["exp2"] + questionnaire["exp5"]

	return questionnaire

# Aggregate answer statistics by each user
def answer_aggregate(ans_train, ans_test, questionnaire, avg_agg = "mean"):
	# Prepare practice time features
	ans_train["train_time_mean"] = ans_train["time"]
	ans_train["train_time_std"] = ans_train["time"]
	ans_train["action_freq"] = (ans_train["cnt_set"] + ans_train["cnt_unset"] + ans_train["cnt_reset"]) / ans_train["time"]
	ans_train["click_time_mean"] = 1. / ans_train["action_freq"]
	ans_train["click_time_std"] = 1. / ans_train["action_freq"]

	# Different aggregation dict, depending on whether student's theta is present
	if "theta" in ans_train:
		agg_dict = {
			"condition": "min",
			"diff": avg_agg,
			"success": "sum",
			"is_timeout": "sum",
			"is_incorrect": "sum",
			"cnt_set": "sum",
			"cnt_unset": "sum",
			"cnt_reset": "sum",
			"cnt_correct": "sum",
			"cnt_incorrect": "sum",
			"cnt_neutral": "sum",
			"theta": "mean",
			"pl-diff": "mean",
			"solve_prob": "mean",
			"adiff": "mean",
			"rdiff": "mean",
			"rdiff_h": "sum",
			"rdiff_m": "sum",
			"rdiff_l": "sum",
			"adiff_h": "sum",
			"adiff_m": "sum",
			"adiff_l": "sum",
			"train_time_mean": "mean",
			"train_time_std": "std",
			"action_freq": "mean",
			"click_time_mean": "mean",
			"click_time_std": "std"
		}
	else:
		agg_dict = {
			"condition": "min",
			"diff": avg_agg,
			"success": "sum",
			"is_timeout": "sum",
			"is_incorrect": "sum",
			"time": avg_agg,
			"cnt_set": "sum",
			"cnt_unset": "sum",
			"cnt_reset": "sum",
			"cnt_correct": "sum",
			"cnt_incorrect": "sum",
			"cnt_neutral": "sum",
			"train_time_mean": "mean",
			"train_time_std": "std",
			"action_freq": "mean",
			"click_time_mean": "mean",
			"click_time_std": "std"
		}

	# Aggregate training outcomes
	ans_agg = ans_train \
		.groupby(by = "user_id") \
		.agg(agg_dict) \
		.rename(columns = {
			"diff": "train_diff",
			"success": "train_success",
			"wrong_select_num": "wsn_train"
		})

	# Aggregate time for each training outcome
	ans_train_success = ans_train.loc[ans_train["success"] == True]
	ans_train_incorrect = ans_train.loc[(ans_train["success"] == False) & (ans_train["time"] < 85)]

	ans_train_success_agg = ans_train_success.groupby(by = "user_id") \
							.agg({ "time": "mean" }) \
							.rename(columns = { "time": "success_time" })
	ans_train_incorrect_agg = ans_train_incorrect.groupby(by = "user_id") \
							.agg({ "time": "mean" }) \
							.rename(columns = { "time": "incorrect_time" })

	# Aggregate test outcomes
	ans_test_agg = ans_test.groupby(by = "user_id") \
							.agg({ "pre": "mean", "post": "mean", "nlg": "mean", "test_diff": "mean" })
	ans_pretest_agg = ans_test.loc[ans_test["phase"] == "pre-test"].groupby(by = "user_id") \
							.agg({ "is_incorrect": "sum", "is_timeout": "sum", "time": "mean" }) \
							.rename(columns = { "is_timeout": "pre-timeout", "is_incorrect": "pre-incorrect", "time": "pre-time" })
	ans_posttest_agg = ans_test.loc[ans_test["phase"] == "post-test"].groupby(by = "user_id") \
							.agg({ "is_incorrect": "sum", "is_timeout": "sum" }) \
							.rename(columns = { "is_timeout": "post-timeout", "is_incorrect": "post-incorrect" })

	ans_agg = ans_agg.join(questionnaire.drop(columns = ["condition", "train_success"]).set_index("user_id"), how = "left", on = "user_id")
	ans_agg = ans_agg.join(ans_test_agg, on = "user_id")
	ans_agg = ans_agg.join(ans_pretest_agg, on = "user_id")
	ans_agg = ans_agg.join(ans_posttest_agg, on = "user_id")
	ans_agg = ans_agg.join(ans_train_success_agg, on = "user_id")
	ans_agg = ans_agg.join(ans_train_incorrect_agg, on = "user_id")

	ans_agg.reset_index(inplace = True)

	return ans_agg

def answer_augment_4pl(ans, thetas):
	ans = ans.copy()

	thetas_pd = pd.DataFrame(list(thetas.items()), columns = ["user_id", "theta"])

	ans = ans.join(thetas_pd.set_index("user_id"), on = "user_id")

	pl_beta = [0.25755704, 6.07254928, -0.33565597, 0.00886291, -0.06892614, 0.04141465, 0.78820168]
	c_solved = 0.0905646
	d_solved = 0.88630125
	ct_solved = 11.04627041

	ans["pl-diff"] = 0.

	for i in itertools.chain(range(1, 192), range(301, 307)):
		ans_graph = ans.loc[ans["chosen_graph"] == i]

		if len(ans_graph) == 0:
			continue

		g_mis = ans_graph.iloc[0]["mis_size"]
		g_prob = ans_graph.iloc[0]["prob"]
		g_vtx = ans_graph.iloc[0]["vertices"]
		g_edges = ans_graph.iloc[0]["edges"]
		g_int = ans_graph.iloc[0]["intersects"]

		g_beta = [g_mis, g_prob, g_vtx, g_edges, g_int, 0, 1]
		g_diff = -np.dot(pl_beta, g_beta)

		ans.loc[ans["chosen_graph"] == i, "pl-diff"] = g_diff

	ans["rdiff"] = ans["pl-diff"] - ct_solved * ans["theta"]
	ans["solve_prob"] = c_solved + (d_solved - c_solved) * sigmoid(ct_solved * ans["theta"] - ans["pl-diff"])

	# Quantiles for relative difficulty
	challenge_l = ans["solve_prob"].quantile([0.33, 0.66]).iloc[0]
	challenge_h = ans["solve_prob"].quantile([0.33, 0.66]).iloc[1]

	ans["rdiff_h"] = challenge_l > ans["solve_prob"]
	ans["rdiff_m"] = (challenge_l <= ans["solve_prob"]) & (ans["solve_prob"] <= challenge_h)
	ans["rdiff_l"] = challenge_h < ans["solve_prob"]

	diff_l = ans["pl-diff"].quantile([0.33, 0.66]).iloc[0]
	diff_h = ans["pl-diff"].quantile([0.33, 0.66]).iloc[1]

	ans["adiff"] = ans["pl-diff"]
	ans["adiff_l"] = diff_l > ans["pl-diff"]
	ans["adiff_m"] = (diff_l <= ans["pl-diff"]) & (ans["pl-diff"] <= diff_h)
	ans["adiff_h"] = diff_h < ans["pl-diff"]

	return ans