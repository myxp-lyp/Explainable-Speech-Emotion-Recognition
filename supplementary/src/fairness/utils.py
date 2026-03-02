from fairness.constants import privileged_groups


def get_unprivileged_groups(df):
    unprivileged_groups = {}
    for col, values in privileged_groups.items():
        all_values = df[col].unique().tolist()
        unprivileged_groups[col] = list(set(all_values) - set(values))

    return unprivileged_groups
