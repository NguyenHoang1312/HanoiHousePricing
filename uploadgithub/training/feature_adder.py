from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def haversine_distance(self, lat1, lon1, lat2=21.0279, lon2=105.8523):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return c * r

    def transform(self, X):
        df = X.copy()

        # Noise injection
        numeric_cols = df.select_dtypes(include='number').columns.to_list()
        for col in numeric_cols:
            sigma = df[col].std() * 0.01
            df[col + '_noised'] = df[col] + np.random.normal(0, sigma, size=len(df))

        if {'chieu_rong', 'chieu_dai'}.issubset(df.columns):
            df['ti_le_dai_rong'] = df['chieu_rong'] / df['chieu_dai']
        if {'vi_do', 'kinh_do'}.issubset(df.columns):
            df['khoang_cach_trung_tam'] = self.haversine_distance(df['vi_do'], df['kinh_do'])
        if {'dien_tich', 'so_tang'}.issubset(df.columns):
            df['do_thong_thoang'] = df['dien_tich'] / df['so_tang']
        return df


