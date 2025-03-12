#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 8600926f2f22e58779a667d82047a90318b20431
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.[all, dev, test]'
git checkout 8600926f2f22e58779a667d82047a90318b20431 statsmodels/stats/tests/test_pairwise.py
git apply -v - <<'EOF_114329324912'
diff --git a/statsmodels/stats/tests/test_pairwise.py b/statsmodels/stats/tests/test_pairwise.py
index 5d34b7a8d11..c9a84ead9fe 100644
--- a/statsmodels/stats/tests/test_pairwise.py
+++ b/statsmodels/stats/tests/test_pairwise.py
@@ -120,6 +120,14 @@
 1 - 3\t-0.260\t-3.909\t3.389\t-
 '''
 
+# result in R: library(rstatix)
+# games_howell_test(df, StressReduction ~ Treatment, conf.level = 0.99)
+ss2_unequal = '''\
+1\tStressReduction\tmedical\tmental\t1.8888888888888888\t0.7123347940930316\t3.0654429836847461\t0.000196\t***
+2\tStressReduction\tmedical\tphysical\t0.8888888888888888\t-0.8105797509636128\t2.5883575287413905\t0.206000\tns
+3\tStressReduction\tmental\tphysical\t-1.0000000000000000\t-2.6647460755237473\t0.6647460755237473\t0.127000\tns
+'''
+
 cylinders = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 6, 6, 6, 4, 4,
                     4, 4, 4, 4, 6, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 6, 6,
                     6, 6, 4, 4, 4, 4, 4, 8, 4, 6, 6, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
@@ -138,6 +146,7 @@
 ss2 = asbytes(ss2)
 ss3 = asbytes(ss3)
 ss5 = asbytes(ss5)
+ss2_unequal = asbytes(ss2_unequal)
 
 dta = pd.read_csv(BytesIO(ss), sep=r'\s+', header=None, engine='python')
 dta.columns = "Rust", "Brand", "Replication"
@@ -151,7 +160,10 @@
 for col in ('pair', 'sig'):
     dta5[col] = dta5[col].map(lambda v: v.encode('utf-8'))
 sas_ = dta5.iloc[[1, 3, 2]]
-
+games_howell_r_result = pd.read_csv(BytesIO(ss2_unequal), sep=r'\t', header=None, engine='python')
+games_howell_r_result.columns = ['idx', 'y', 'group1', 'group2', 'meandiff', 'lower', 'upper', 'pvalue', 'sig']
+for col in ('y', 'group1', 'group2', 'sig'):
+    games_howell_r_result[col] = games_howell_r_result[col].map(lambda v: v.encode('utf-8'))
 
 def get_thsd(mci, alpha=0.05):
     var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
@@ -172,7 +184,10 @@ class CheckTuckeyHSDMixin:
     @classmethod
     def setup_class_(cls):
         cls.mc = MultiComparison(cls.endog, cls.groups)
-        cls.res = cls.mc.tukeyhsd(alpha=cls.alpha)
+        if hasattr(cls, 'use_var'):
+            cls.res = cls.mc.tukeyhsd(alpha=cls.alpha, use_var=cls.use_var)
+        else:
+            cls.res = cls.mc.tukeyhsd(alpha=cls.alpha)
 
     def test_multicomptukey(self):
         assert_almost_equal(self.res.meandiffs, self.meandiff2, decimal=14)
@@ -180,12 +195,18 @@ def test_multicomptukey(self):
         assert_equal(self.res.reject, self.reject2)
 
     def test_group_tukey(self):
+        if hasattr(self, 'use_var') and self.use_var == 'unequal':
+            # in unequal variance case, we feed groupvarwithin, no need to test total variance
+            return
         res_t = get_thsd(self.mc, alpha=self.alpha)
         assert_almost_equal(res_t[4], self.confint2, decimal=2)
 
     def test_shortcut_function(self):
         #check wrapper function
-        res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
+        if hasattr(self, 'use_var'):
+            res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha, use_var=self.use_var)
+        else:
+            res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
         assert_almost_equal(res.confint, self.res.confint, decimal=14)
 
     @pytest.mark.smoke
@@ -244,6 +265,15 @@ def test_table_names_custom_group_order(self):
             second_group = t[i][1].data
             assert_((first_group, second_group) == expected_order[i - 1])
 
+        frame = res.summary_frame()
+        assert_equal(frame["p-adj"], res.pvalues)
+        assert_equal(frame["meandiff"], res.meandiffs)
+        # Why are we working with binary strings, old time numpy?
+        group_t = [b"medical", b"mental", b"mental"]
+        group_c = [b"physical", b"physical", b"medical"]
+        assert frame["group_t"].to_list() == group_t
+        assert frame["group_c"].to_list() == group_c
+
 
 class TestTuckeyHSD2Pandas(TestTuckeyHSD2):
 
@@ -323,6 +353,23 @@ def setup_class(cls):
         cls.reject2 = pvals < 0.01
 
 
+class TestTukeyHSD2sUnequal(CheckTuckeyHSDMixin):
+
+    @classmethod
+    def setup_class(cls):
+        # Games-Howell test
+        cls.endog = dta2['StressReduction'][3:29]
+        cls.groups = dta2['Treatment'][3:29]
+        cls.alpha = 0.01
+        cls.use_var = 'unequal'
+        cls.setup_class_()
+
+        #from R: library(rstatix)
+        cls.meandiff2 = games_howell_r_result['meandiff']
+        cls.confint2 = games_howell_r_result[['lower', 'upper']].astype(float).values.reshape((3, 2))
+        cls.reject2 = games_howell_r_result['sig'] == asbytes('***')
+
+
 class TestTuckeyHSD3(CheckTuckeyHSDMixin):
 
     @classmethod
@@ -365,3 +412,18 @@ def setup_class(cls):
 
     def test_hochberg_intervals(self):
         assert_almost_equal(self.res.halfwidths, self.halfwidth2, 4)
+
+
+@pytest.mark.smoke
+@pytest.mark.matplotlib
+def test_plot(close_figures):
+    # SMOKE test
+    cylinders_adj = cylinders.astype(float)
+    # avoid zero division, zero within variance in France and Sweden
+    cylinders_adj[[10, 28]] += 0.05
+    alpha = 0.05
+    mc = MultiComparison(cylinders_adj, cyl_labels)
+    resth = mc.tukeyhsd(alpha=alpha, use_var="equal")
+    resgh = mc.tukeyhsd(alpha=alpha, use_var="unequal")
+    resth.plot_simultaneous()
+    resgh.plot_simultaneous()

EOF_114329324912
pytest -rA --tb=long statsmodels/stats/tests/test_pairwise.py
git checkout 8600926f2f22e58779a667d82047a90318b20431 statsmodels/stats/tests/test_pairwise.py
