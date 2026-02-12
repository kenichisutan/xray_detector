import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator

THR = np.arange(0, 501, 2, dtype=np.float64)
CHAOTIC_END_MIN = 67
CHAOTIC_END_MAX = 90
SAFE_MARGIN = 2

ENS_SG = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]
ENS_GAUSS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
ENS_CW = [4, 5, 6]
FALL_OFFSETS = [0, 1, 2]
WINSOR_P = 0.05
MEDIAN_K = 9
N_KEEP = 480
BIAS_SCALE = 0.84
BIAS_EXT_SCALE = 1.0
NOISE_THRESH = 200
BIAS_BIN_EDGES = [150, 170, 185, 200, 210, 220, 230, 240, 250, 500]

Q5_A = np.array([2., -1., -2., -1., 2.]) / 14.0
Q5_B = np.array([-2., -1., 0., 1., 2.]) / 10.0
Q7_A = np.array([5., 0., -3., -4., -3., 0., 5.]) / 84.0
Q7_B = np.array([-3., -2., -1., 0., 1., 2., 3.]) / 28.0
Q9_A = np.array([28., 7., -8., -17., -20., -17., -8., 7., 28.]) / 924.0
Q9_B = np.array([-4., -3., -2., -1., 0., 1., 2., 3., 4.]) / 60.0


class Model(BaseEstimator):
    def __init__(self):
        self.fall_start = 75
        self.selected = None
        self.bias_table = []
        self.adapt_median = 255.0
        self.bias_poly = None

    def _detect_fall_start(self, X):
        X = np.asarray(X, dtype=np.float64)
        avg_curve = X.mean(axis=0)
        abs_diff = np.abs(np.diff(avg_curve))
        candidate = abs_diff[CHAOTIC_END_MIN:CHAOTIC_END_MAX]
        if len(candidate) == 0:
            return 75
        smooth_diff = gaussian_filter1d(candidate, sigma=2)
        latter_half = smooth_diff[len(smooth_diff) // 2:]
        if len(latter_half) == 0:
            return 75
        threshold = max(np.median(latter_half) * 5, 100)
        for i in range(len(smooth_diff) - 2):
            if all(smooth_diff[i + j] < threshold for j in range(3)):
                return max(CHAOTIC_END_MIN + i + SAFE_MARGIN,
                           CHAOTIC_END_MIN + SAFE_MARGIN)
        return 75

    def _batch_com(self, deriv, fall_x, peaks, cw):
        n, n_cols = deriv.shape
        wsize = 2 * cw + 1
        padded = np.pad(deriv, ((0, 0), (cw, cw)),
                        mode='constant', constant_values=0)
        ext_x = fall_x[0] - 2.0 * cw + 2.0 * np.arange(n_cols + 2 * cw)
        col_idx = peaks[:, None] + np.arange(wsize)
        row_idx = np.arange(n)[:, None]
        windows = padded[row_idx, col_idx]
        x_windows = ext_x[col_idx]
        w = np.maximum(windows, 0)
        w_sum = w.sum(axis=1)
        safe_wsum = np.where(w_sum > 1e-12, w_sum, 1.0)
        fallback = fall_x[np.minimum(peaks, n_cols - 1)]
        return np.where(w_sum > 1e-12,
                        (w * x_windows).sum(axis=1) / safe_wsum, fallback)

    def _batch_quad(self, deriv, fall_x, peaks, hw, q_a, q_b):
        n, n_cols = deriv.shape
        padded = np.pad(deriv, ((0, 0), (hw, hw)), mode='edge')
        idx = peaks[:, None] + np.arange(2 * hw + 1)
        windows = padded[np.arange(n)[:, None], idx]
        a_coeff = windows @ q_a
        b_coeff = windows @ q_b
        valid = a_coeff < -1e-12
        safe_a = np.where(valid, a_coeff, -1.0)
        t_star = np.where(valid, -b_coeff / (2 * safe_a), 0.0)
        t_star = np.clip(t_star, -hw + 0.5, hw - 0.5)
        base_dac = fall_x[np.minimum(peaks, n_cols - 1)]
        return base_dac + t_star * 2.0

    def _batch_loggauss(self, deriv, fall_x, peaks, hw, q_a, q_b):
        n, n_cols = deriv.shape
        padded = np.pad(deriv, ((0, 0), (hw, hw)), mode='edge')
        idx = peaks[:, None] + np.arange(2 * hw + 1)
        windows = padded[np.arange(n)[:, None], idx]
        all_pos = (windows > 1e-6).all(axis=1)
        safe_win = np.maximum(windows, 1e-6)
        lw = np.log(safe_win)
        a_coeff = lw @ q_a
        b_coeff = lw @ q_b
        valid = all_pos & (a_coeff < -1e-12)
        safe_a = np.where(valid, a_coeff, -1.0)
        t_star = np.where(valid, -b_coeff / (2 * safe_a), 0.0)
        t_star = np.clip(t_star, -hw + 0.5, hw - 0.5)
        base_dac = fall_x[np.minimum(peaks, n_cols - 1)]
        return np.where(valid, base_dac + t_star * 2.0, base_dac)

    def _add_estimates(self, est_list, deriv, fall_x, peaks):
        for cw in ENS_CW:
            est_list.append(self._batch_com(deriv, fall_x, peaks, cw))
        est_list.append(self._batch_quad(deriv, fall_x, peaks, 2, Q5_A, Q5_B))
        est_list.append(self._batch_quad(deriv, fall_x, peaks, 4, Q9_A, Q9_B))
        est_list.append(
            self._batch_loggauss(deriv, fall_x, peaks, 2, Q5_A, Q5_B))
        est_list.append(
            self._batch_loggauss(deriv, fall_x, peaks, 3, Q7_A, Q7_B))
        est_list.append(
            self._batch_loggauss(deriv, fall_x, peaks, 4, Q9_A, Q9_B))

    def _collect_all(self, X):
        all_estimates = []
        for offset in FALL_OFFSETS:
            fs = max(CHAOTIC_END_MIN + SAFE_MARGIN,
                     self.fall_start + offset)
            if fs >= X.shape[1] - 10:
                continue
            X_fall = X[:, fs:]
            fall_x = THR[fs:]
            n_cols = X_fall.shape[1]

            for sgw in ENS_SG:
                if n_cols < sgw or n_cols < 10:
                    continue
                deriv = -savgol_filter(X_fall, window_length=sgw,
                                       polyorder=3, deriv=1,
                                       delta=2.0, axis=1)
                peaks = np.argmax(deriv, axis=1)
                self._add_estimates(all_estimates, deriv, fall_x, peaks)

            for sig in ENS_GAUSS:
                if n_cols < 10:
                    continue
                smoothed = gaussian_filter1d(X_fall, sigma=sig, axis=1)
                deriv = -np.gradient(smoothed, 2.0, axis=1)
                peaks = np.argmax(deriv, axis=1)
                self._add_estimates(all_estimates, deriv, fall_x, peaks)

        return np.array(all_estimates)

    def _aggregate(self, E):
        if self.selected is not None:
            E = E[self.selected]
        lo_val = np.percentile(E, WINSOR_P * 100, axis=0)
        hi_val = np.percentile(E, (1 - WINSOR_P) * 100, axis=0)
        clipped = np.clip(E, lo_val[None, :], hi_val[None, :])
        return clipped.mean(axis=0)

    def _predict_raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_med = median_filter(X, size=(1, MEDIAN_K))
        E = self._collect_all(X_med)
        return self._aggregate(E)

    def _select_estimators(self, X_adapt):
        X_ad = median_filter(X_adapt, size=(1, MEDIAN_K)).astype(np.float64)
        E_ad = self._collect_all(X_ad)
        n_est = E_ad.shape[0]
        consensus = self._aggregate(E_ad)
        self.adapt_median = float(np.median(consensus))
        dev = np.array([np.mean((E_ad[i] - consensus) ** 2)
                        for i in range(n_est)])
        order = np.argsort(dev)
        n_keep = min(N_KEEP, n_est)
        self.selected = np.sort(order[:n_keep])
        print(f"  Selected {n_keep}/{n_est} estimators by adapt-data consistency")

    def _calibrate_bias(self, X_train, y_train):
        y_ens = self._predict_raw(X_train)
        residuals = y_train - y_ens
        self.bias_table = []
        for i in range(len(BIAS_BIN_EDGES) - 1):
            lo, hi = BIAS_BIN_EDGES[i], BIAS_BIN_EDGES[i + 1]
            mask = (y_ens >= lo) & (y_ens < hi) & (np.abs(residuals) < 10)
            if mask.sum() >= 5:
                self.bias_table.append(
                    (lo, hi, float(residuals[mask].mean())))
        keep = np.abs(residuals) < 10
        from numpy.polynomial import polynomial as P
        self.bias_poly = P.polyfit(y_ens[keep], residuals[keep], 2)
        print(f"  Bias calibration: {len(self.bias_table)} bins + poly2 extrapolation")

    def fit(self, X_train, y_train, X_adapt=None):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        has_da = X_adapt is not None and len(X_adapt) > 0
        if has_da:
            X_adapt = np.asarray(X_adapt, dtype=np.float64)
            fs_adapt = self._detect_fall_start(X_adapt)
            fs_source = self._detect_fall_start(X_train)
            self.fall_start = max(fs_adapt, fs_source)
            print(f"  fall_start: src={fs_source} adp={fs_adapt} "
                  f"-> {self.fall_start}")
            self._select_estimators(X_adapt)
        else:
            self.fall_start = self._detect_fall_start(X_train)
            self.selected = None
            print(f"  fall_start: {self.fall_start}")

        self._calibrate_bias(X_train, y_train)

        n_per = len(ENS_CW) + 5
        n_methods = len(ENS_SG) + len(ENS_GAUSS)
        n_total = len(FALL_OFFSETS) * n_methods * n_per
        n_used = len(self.selected) if self.selected is not None else n_total
        print(f"  Ensemble: {n_used}/{n_total} estimators, "
              f"winsorized mean [{int(WINSOR_P*100)}%]")
        print("Training complete")

    def predict(self, X_test):
        X = np.asarray(X_test, dtype=np.float64)
        noise = np.std(np.diff(X[:, 75:], axis=1), axis=1)
        X_med = median_filter(X, size=(1, MEDIAN_K))
        E = self._collect_all(X_med)
        y_pred = self._aggregate(E)
        y_raw = y_pred.copy()

        noise_mask = noise > NOISE_THRESH
        y_pred[noise_mask] = self.adapt_median

        for lo, hi, bias in self.bias_table:
            mask = (y_raw >= lo) & (y_raw < hi) & (~noise_mask)
            y_pred[mask] += BIAS_SCALE * bias

        from numpy.polynomial import polynomial as P
        src_max = BIAS_BIN_EDGES[-2]
        ext_mask = (y_raw >= src_max) & (~noise_mask)
        if ext_mask.any():
            y_pred[ext_mask] += (BIAS_EXT_SCALE * BIAS_SCALE
                                 * P.polyval(y_raw[ext_mask], self.bias_poly))

        return y_pred
