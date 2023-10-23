import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pingouin as pg
import random
import scipy.special as special
import scipy.stats as stats
import seaborn as sns
import sys

from efficient_apriori import apriori
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

import data_utils as d_utils

clustering_class = KMeans
clustering_params = {				# KMeans params
	"random_state": 1010,
	"n_init": 100,
	"max_iter": 500
}
elbow_method = True

if len(sys.argv) == 1:
	print("Please specify working directory")
	sys.exit(0)
else:
	settings_dir = sys.argv[1]

	with open("plots/%s/settings.json" % settings_dir, "r") as fp:
		settings = json.load(fp)

	label_cols = settings["label_cols"]
	label_names = settings["label_names"]
	label_ranges = settings["label_ranges"]
	plot_clusters_x = settings["plot_clusters_x"]
	plot_types = settings["plot_types"]

	norm_columns = settings["norm_columns"]
	cols_cluster = settings["cols_cluster"]
	ratio_groups = settings["ratio_groups"]

	save_folder = "plots/%s/" % settings_dir

logs = []

def plot_elbow_method(data, filename = "plots/elbow.png"):
	sse = []

	for n_clusters in range(2, 16):
		kmeans = clustering_class(n_clusters = n_clusters, **clustering_params).fit(data)
		sse.append(kmeans.inertia_)

	plt.plot(list(range(2, 16)), sse)
	plt.xlabel("Number of clusters")
	plt.ylabel("Inertia")
	plt.title("Elbow method")
	plt.savefig(filename, bbox_inches = "tight", dpi = 300)

	kl = KneeLocator(range(2, 16), sse, curve = "convex", direction = "decreasing")
	print("Elbow at %d" % kl.elbow)

	return kl.elbow

def plot_clusters(data, n_clusters, fit_columns, label_column, label_name, label_range, filename = None, blobs = None, plot_type = "bar", plot_clusters_x = None):
	global logs
	data = data.copy()

	sig_threshold = 0.05

	x_fit = clustering_class(n_clusters = n_clusters, **clustering_params).fit_predict(data[fit_columns])
	y = data[label_column].tolist()

	data["x_fit"] = x_fit

	# Check difference between clusters by Kruskal-Wallis
	cluster_values = [[] for _ in range(n_clusters)]

	for xe, ye in zip(x_fit, y):
		cluster_values[xe].append(ye)

	stat, p = stats.kruskal(*cluster_values)

	# Eta-squared effect size
	eta_sq = (stat - n_clusters + 1) / (len(y) - n_clusters)

	print("%s Normality" % label_name, stats.shapiro(y))
	print("%s Bartlett" % label_name, stats.bartlett(*cluster_values))
	print("%s ANOVA" % label_name, stats.f_oneway(*cluster_values))
	print(round(pg.anova(dv = label_column, between = "x_fit", data = data, detailed = True, effsize='n2'), 4))
	print("%s Kruskal-Wallis" % label_name, "H-stat = %.2f, p = %.3f, eta-squared = %.3f" % (stat, p, eta_sq))
	print("Member in clusters", [len(x) for x in cluster_values])

	logs.append({
		"type": "kruskal-wallis",
		"label_col": label_column,
		"p": p,
		"eta_sq": eta_sq,
		"h_stat": stat,
		"means": [np.mean(cluster_values[i]) for i in range(n_clusters)],
		"stds": [np.std(cluster_values[i]) for i in range(n_clusters)],
		"cnts": [len(x) for x in cluster_values]
	})

	pairwise_tests = [[""] * n_clusters for _ in range(n_clusters)]
	pairwise_stats = [[0.] * n_clusters for _ in range(n_clusters)]
	pairwise_ps = [[0.] * n_clusters for _ in range(n_clusters)]
	pairwise_es = [[0.] * n_clusters for _ in range(n_clusters)]
	pairwise_sigs = [[" "] * n_clusters for _ in range(n_clusters)]
	benjamini_ps = []
	sigdiff_pairs = []

	# Pairwise tests between the clusters
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			# Normality tests to determine test type
			if len(cluster_values[i]) >= 3:
				i_normality_p = stats.shapiro(cluster_values[i]).pvalue
			else:
				i_normality_p = 1.

			if len(cluster_values[j]) >= 3:
				j_normality_p = stats.shapiro(cluster_values[j]).pvalue
			else:
				j_normality_p = 1.

			if i_normality_p < sig_threshold or j_normality_p < sig_threshold:
				pairwise_tests[i][j] = "mann-whitney"
				test_fx = stats.mannwhitneyu
			else:
				pairwise_tests[i][j] = "t-test"
				test_fx = stats.ttest_ind
			
			pairwise_stats[i][j] = test_fx(cluster_values[i], cluster_values[j]).statistic
			pairwise_ps[i][j] = test_fx(cluster_values[i], cluster_values[j]).pvalue
			benjamini_ps.append((pairwise_ps[i][j], (i, j)))

			if pairwise_tests[i][j] == "mann-whitney":
				# Report PS as effect size
				pairwise_es[i][j] = pairwise_stats[i][j] / len(cluster_values[i]) / len(cluster_values[j])

				if pairwise_es[i][j] < 0.5:
					pairwise_es[i][j] = 1. - pairwise_es[i][j]
			elif pairwise_tests[i][j] == "t-test":
				# Report Cohen's d as effect size
				pairwise_es[i][j] = (np.mean(cluster_values[i]) - np.mean(cluster_values[j])) / np.std(cluster_values[i] + cluster_values[j])

	# Figure out significance with Benjamini-Hochberg
	benjamini_ps.sort(key = lambda x: x[0])
	for i, (p, (cluster_i, cluster_j)) in enumerate(benjamini_ps):
		if p * (i + 1) < sig_threshold:
			pairwise_sigs[cluster_i][cluster_j] = "*"
			pairwise_sigs[cluster_j][cluster_i] = "*"
			sigdiff_pairs.append((cluster_i, cluster_j))
		else:
			break

	logs.append({
		"type": "pairwise",
		"label_col": label_column,
		"tests": pairwise_tests,
		"statistics": pairwise_stats,
		"effect_size": pairwise_es,
		"p": pairwise_ps,
		"sig": pairwise_sigs
	})

	for i in range(n_clusters):
		print("Cluster %d = %.4f +- %.4f" % (i, np.mean(cluster_values[i]), np.std(cluster_values[i])))
		
	# Box plot by each cluster
	# If clusters to plot is not stated, plot all clusters
	if plot_clusters_x is None:
		plot_clusters_x = list(range(n_clusters))

	df = pd.DataFrame({ "cluster": x_fit, "label": y })
	plt.clf()
	sns.set(rc = { "figure.figsize": (1.619, 1) })

	if plot_type == "bar":
		sns.barplot(data = df, x = "cluster", y = "label", palette = "colorblind", order = plot_clusters_x)
	elif plot_type == "box":
		sns.boxplot(data = df, x = "cluster", y = "label", palette = "colorblind", medianprops = {'color': 'red', 'lw': 2}, order = plot_clusters_x)

	for i, sd in enumerate(sigdiff_pairs):
	    x1 = sd[0]
	    x2 = sd[1]
	    # Plot the bar
	    bar_height = 1 + 2 * i * 0.06
	    bar_tips = bar_height - 2 * 0.02
	    plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
	    # Significance level
	    text_height = bar_height - 0.03
	    plt.text((x1 + x2) * 0.5, text_height, "*", ha="center", va="bottom", c='k')

	plt.xlabel("Clusters")
	plt.ylabel(label_name)

	if label_range is not None:
		plt.ylim(label_range)

	plt.title("%s of each cluster" % label_name)
	plt.savefig(filename, bbox_inches = "tight", dpi = 300)

def chi2_clusters(data, n_clusters, fit_columns):
	global logs

	condition_idxs = {
		"predef": 0,
		"self-det": 1,
		"dda": 2
	}

	x_fit = clustering_class(n_clusters = n_clusters, **clustering_params).fit_predict(data[fit_columns])

	# Check difference between clusters by Kruskal-Wallis
	condition_cluster_cnts = [[0] * n_clusters for _ in range(3)]

	for xe, condition in zip(x_fit, data["condition"]):
		condition_cluster_cnts[condition_idxs[condition]][xe] += 1

	res = stats.chi2_contingency(condition_cluster_cnts) 

	print(condition_cluster_cnts)

	print("Condition-cluster Chi-2", "p =", res[1], ", test statistic = ", res[0], ", dof = ", res[2])

	logs.append({
		"type": "condition-cluster-chi2",
		"p": res[1],
		"chi2": res[0],
		"dof": res[2],
		"condition-cluster-cnts": condition_cluster_cnts
	})

def association_rule_mining(data, n_clusters, fit_columns, rule_columns, min_cluster_support = 0.5, min_confidence = 0.5, log_type = "rules"):
	global logs

	rules_extracted = []

	x_fit = clustering_class(n_clusters = n_clusters, **clustering_params).fit_predict(data[fit_columns])

	cluster_sizes = [0] * n_clusters

	for xe in x_fit:
		cluster_sizes[xe] += 1

	baskets = [["cluster-%d" % x] for x in x_fit]

	for rule_column in rule_columns:
		if rule_column in []:
			data_binneds = KBinsDiscretizer(n_bins = 4, encode = "ordinal", strategy = "kmeans").fit_transform(data[[rule_column]])
			discrete_labels = ["l", "m", "h", "vh"]
		else:
			data_binneds = KBinsDiscretizer(n_bins = 3, encode = "ordinal", strategy = "kmeans").fit_transform(data[[rule_column]])
			discrete_labels = ["l", "m", "h"]

		for i, data_binned in enumerate(data_binneds):
			baskets[i].append("%s-%s" % (rule_column, discrete_labels[int(data_binned[0])]))

	for cluster in range(n_clusters):
		min_support = min_cluster_support * cluster_sizes[cluster] / sum(cluster_sizes)

		itemsets, rules = apriori(baskets, min_support = min_support, min_confidence = min_confidence, max_length = 5)

		rules_rhs = filter(lambda rule: len(rule.lhs) <= 4 and len(rule.rhs) == 1, rules)			# Rules with limited size on LHS and RHS
		rules_rhs = filter(lambda rule: rule.rhs[0] == "cluster-%d" % cluster, rules_rhs)			# Rules for current cluster only
		rules_rhs = sorted(rules_rhs, key = lambda rule: rule.conviction)

		for rule in rules_rhs:
			if rule.rhs[0][:7] == "cluster":
				print(rule)
				rules_extracted.append({
					"lhs": rule.lhs,
					"rhs": rule.rhs[0],
					"cluster_supp": rule.support / (cluster_sizes[cluster] / sum(cluster_sizes)),
					"confidence": rule.confidence,
					"lift": rule.lift
				})

	logs.append({
		"type": log_type,
		"rules": rules_extracted
	})

def mahalanobis_dist(y = None, data = None, cov = None):
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

ans, questionnaire, thetas = d_utils.load_data(suffix = "unfiltered")

ans, ans_train, ans_test = d_utils.answer_preprocess(ans)
ans_train = d_utils.answer_augment_4pl(ans_train, thetas)
questionnaire = d_utils.questionnaire_preprocess(questionnaire)
ans_agg = d_utils.answer_aggregate(ans_train, ans_test, questionnaire)
ans_click_times = pd.read_csv("data/click_times.csv")

ans_agg = pd.merge(ans_agg, ans_click_times, how = "inner", on = ["user_id"])

# Store all raw values before normalisation
for norm_column in norm_columns:
	ans_agg.loc[:, norm_column + "_raw"] = ans_agg[norm_column]

for ratio_group in ratio_groups:
	for ratio_col in ratio_group:
		ans_agg.loc[:, ratio_col + "_raw"] = ans_agg[ratio_col]

print("Input data contains %d rows" % len(ans_agg))

# Remove rows with a nan
ans_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
ans_agg = ans_agg.loc[~ans_agg[cols_cluster + label_cols].isnull().any(axis = 1)]

print("NA removal: %d rows remain" % len(ans_agg))

# Remove outliers with Mahalanobis distance
mahalanobis_p_threshold = 0.01
ans_agg["mahalanobis"] = mahalanobis_dist(y = ans_agg[cols_cluster], data = ans_agg[cols_cluster])
ans_agg["mahalanobis_p"] = 1 - stats.chi2.cdf(ans_agg["mahalanobis"], 3)
ans_agg = ans_agg.loc[ans_agg["mahalanobis_p"] >= mahalanobis_p_threshold]

print("Mahalanobis distance outliers: %d rows remain" % len(ans_agg))

# Transfer aggregated results onto single train events
for i in range(len(ans_agg)):
	for label_col in label_cols:
		ans_train.loc[ans_train["user_id"] == ans_agg.iloc[i]["user_id"], label_col] = ans_agg.iloc[i][label_col]

# Normalise data
# Turn count columns into user-wise percentages
for ratio_group in ratio_groups:
	ratio_col_names = [ratio_col + "_ratio" for ratio_col in ratio_group]

	cols_sum = np.sum(ans_agg[ratio_group].to_numpy(), axis = 1)
	ans_agg.loc[:, ratio_col_names] = (ans_agg[ratio_group] / np.expand_dims(cols_sum, 1)).rename(dict(list(zip(ratio_group, ratio_col_names))), axis = "columns")

# Normalise everything by mean-std
for norm_column in norm_columns:
	col_min = np.min(ans_agg[norm_column].tolist())
	col_max = np.max(ans_agg[norm_column].tolist())

	ans_agg.loc[:, norm_column] -= col_min
	ans_agg.loc[:, norm_column] /= ((col_max - col_min) + 1e-6)

# Prepare folder to save stuff
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

# Plot each cluster with corresponding label
if elbow_method:
	n_clusters = plot_elbow_method(ans_agg[cols_cluster], filename = "%s/elbow.png" % save_folder)
else:
	n_clusters = 3

for label_col, label_name, label_range, plot_type in zip(label_cols, label_names, label_ranges, plot_types):
	plot_clusters(ans_agg, n_clusters, cols_cluster, label_col, label_name, label_range, "%s/%s.png" % (save_folder, label_col), plot_type = plot_type, plot_clusters_x = plot_clusters_x)

chi2_clusters(ans_agg, n_clusters, cols_cluster)

# Find the association rules for the clusters
association_rule_mining(ans_agg[cols_cluster], n_clusters, cols_cluster, cols_cluster, log_type = "rules_fit_columns", min_cluster_support = 0.5)

with open("%s/settings.json" % save_folder, "w") as fp:
	json.dump({
		"norm_columns": norm_columns,
		"cols_cluster": cols_cluster,
		"ratio_groups": ratio_groups,
		"label_cols": label_cols,
		"label_names": label_names,
		"plot_types": plot_types,
		"label_ranges": label_ranges,
		"plot_clusters_x": plot_clusters_x
	}, fp, indent = 4)

with open("%s/log.json" % save_folder, "w") as fp:
	json.dump(logs, fp, indent = 4)