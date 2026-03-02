# Train dummy regressor as a basline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_mean_regressor(waf_dataset):

    dummy = DummyRegressor(strategy="mean")
    x_train = waf_dataset["waf_features"]
    y_train = waf_dataset["waf_label"]

    dummy.fit(x_train, y_train)

    y_dummy_pred = dummy.predict(x_train)

    # Evaluate
    mse_dummy = mean_squared_error(y_train, y_dummy_pred)
    r2_dummy = r2_score(y_train, y_dummy_pred)

    return mse_dummy, r2_dummy
