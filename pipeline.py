import pandas as pd
import numpy as np
import dill
import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score



def delete_columns(df):
    columns_to_drop = ['device_model', 'utm_keyword', 'device_os',
                       'event_value', 'hit_time', 'hit_referer', 'event_label' ]
    df = df.drop(columns=columns_to_drop, axis=1).copy()
    return df

def input_values(df):
    df["device_brand"] = df["device_brand"].fillna('other')
    df["utm_adcontent"] = df["utm_adcontent"].fillna('other')
    df["utm_campaign"] = df["utm_campaign"].fillna('other')
    df["utm_source"] = df["utm_source"].fillna('other')
    df = df.copy()
    return df

def transform_features(df, pd=pd):

    city_counts = df['geo_city'].value_counts()
    country_counts = df['geo_country'].value_counts()
    utm_campaign_counts = df['utm_campaign'].value_counts()
    utm_adcontent_counts = df['utm_adcontent'].value_counts()
    device_brand_counts = df['device_brand'].value_counts()

    df['geo_city'] = df['geo_city'].apply(
        lambda x: 'other_cities' if city_counts.get(x, 0) <= 1 else x)
    df['geo_country'] = df['geo_country'].apply(
        lambda x: 'other_countries' if country_counts.get(x, 0) <= 44 else x)
    df['utm_campaign'] = df['utm_campaign'].apply(
        lambda x: 'other_utm_campaign' if utm_campaign_counts.get(x, 0) <= 8 else x)
    df['utm_adcontent'] = df['utm_adcontent'].apply(
        lambda x: 'other_utm_adcontent' if utm_adcontent_counts.get(x, 0) <= 7 else x)
    df['device_brand'] = df['device_brand'].apply(
        lambda x: 'other_device_brand' if device_brand_counts.get(x, 0) <= 2 else x)

    df["visit_time"] = pd.to_datetime(df.visit_time)
    df['vist_hours'] = df["visit_time"].dt.hour
    df = df.drop(columns='visit_time')

    df['hit_page_path'] = df.hit_page_path.apply(lambda x: x.split('/')[0])
    hit_page_path_counts = df['hit_page_path'].value_counts()

    df.hit_page_path = df.hit_page_path.apply(
        lambda x: 'other_hit_page_path' if hit_page_path_counts[x] <= 8 else x)


    df['resoluton1'] = df["device_screen_resolution"].apply(
        lambda x: x.split('x')[0])
    df['resoluton2'] = df["device_screen_resolution"].apply(
        lambda x: x.split('x')[1])

    resoluton1_counts = df['resoluton1'].value_counts()
    resoluton2_counts = df['resoluton2'].value_counts()

    df['resoluton1'] = df['resoluton1'].apply(
        lambda x: 'other_resoluton1' if resoluton1_counts[x] <= 1 else x)
    df['resoluton2'] = df['resoluton2'].apply(
        lambda x: 'other_resoluton2' if resoluton2_counts[x] <= 1 else x)

    df = df.drop(columns='device_screen_resolution')

    df['hit_date'] = pd.to_datetime(df['hit_date'])
    df['hit_month'] = df['hit_date'].dt.month
    df['hit_day'] = df['hit_date'].dt.day
    df = df.drop(columns=['hit_date'])

    df = df.copy()

    return df


class TargetActionPrediction:
    def __init__(self, preprocessor, model, target_column='event_action'):
        self.preprocessor = preprocessor
        self.model = model
        self.target_column = target_column

    def fit(self, X_raw):
        # y_raw: DataFrame с id и flag
        # X_raw: полный raw (26 млн строк)

        cat_features = ['utm_source', 'utm_medium', 'utm_campaign',
                       'utm_adcontent', 'device_category', 'device_brand', 'device_browser',
                       'geo_country', 'geo_city', 'hit_type', 'hit_page_path',
                       'event_category', 'vist_hours', 'resoluton1',
                       'resoluton2', 'hit_month', 'hit_day']

        X_processed = self.preprocessor.fit_transform(X_raw)

        X_final = X_processed.drop(columns=[self.target_column])
        y_final = X_processed[self.target_column]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        roc_auc_scores = []
        n_splits = kf.get_n_splits(X_final, y_final)

        for fold_num, (train_index, val_index) in enumerate(kf.split(X_final, y_final)):
            X_train, X_test = X_final.iloc[train_index], X_final.iloc[val_index]
            y_train, y_test = y_final.iloc[train_index], y_final.iloc[val_index]

            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
                X_train, y_train, random_state=42, stratify=y_train
            )

            self.model.fit(X_train_val, y_train_val, eval_set=(X_test_val, y_test_val), cat_features=cat_features)

            y_pred = self.model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            roc_auc_scores.append(auc)
            print(f"Fold {fold_num + 1} AUC: {auc:.4f}")

            if fold_num == n_splits - 1:
                results_df = pd.DataFrame({
                    "true_flag": y_test.values,
                    "proba": y_pred
                }).sort_values(by="proba", ascending=False).reset_index(drop=True)
                results_df.to_csv("predictions.csv", index=False)

        print(f'Mean AUC: {np.mean(roc_auc_scores):.4f}, Std: {np.std(roc_auc_scores):.4f}')


        self.model.fit(X_final, y_final, cat_features=cat_features)
        return self



def main():

    df = pd.read_csv('data/df_ready.csv')

    target_values = [
        'sub_car_claim_click', 'sub_car_claim_submit_click',
        'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]

    df['event_action'] = df['event_action'].apply(lambda x: 1 if x in target_values else 0)


    preprocessor = Pipeline(steps=[
        ('delete_columns', FunctionTransformer(delete_columns)),
        ('input_values', FunctionTransformer(input_values)),
        ('transform_features', FunctionTransformer(transform_features)),
    ])

    model = CatBoostClassifier(
        loss_function='Logloss',
        random_seed=42,
        verbose=50,
        class_weights={0: 1, 1: 10},
        eval_metric='AUC',
        bagging_temperature=0.9699098521619943,
        border_count=236,
        depth=9,
        iterations=47,
        l2_leaf_reg=2,
        learning_rate=0.028182496720710062,
        random_strength=0.18340450985343382
    )

    main_pipe = TargetActionPrediction(preprocessor, model)
    main_pipe.fit(df)


    with open('target_action_prediction_model.pkl', 'wb') as file:
        dill.dump({
            'model': main_pipe,
            'metadata': {
                'name': 'target_action_prediction',
                'author': 'Artjom Zaicev',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(main_pipe).__name__,
            }
        }, file)

if __name__ == '__main__':
    main()
