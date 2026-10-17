import pandas as pd

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

wide = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv', sep='\t') 

cols = [f'loop_name_{cl}' for cl in CELL_LINES]
able_cols = [f'AbLE_score_{cl}' for cl in CELL_LINES]
sub = wide[cols + able_cols].dropna() 
sub = sub.loc[(sub[able_cols] != 0).all(axis=1)]
sub = sub[cols]

same = (sub.nunique(axis=1) == 1)

print(f"n loops compared (present at all {len(CELL_LINES)} stages): {len(sub):,}")
print(f"same class throughout:    {same.sum():,}  ({same.mean()*100:.1f}%)")
print(f"class changed at least once: {(~same).sum():,}  ({(~same).mean()*100:.1f}%)")

print("\nbreakdown of changed loops, by (first stage class -> last stage class):")
changed = sub.loc[~same]
print(
    changed.groupby([cols[0], cols[-1]]).size()
    .sort_values(ascending=False)
    .rename('n').to_string()
)