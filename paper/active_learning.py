from carbondriver import GDEOptimizer
from carbondriver.loaders import load_data

X, y, means, stds, df = load_data()
df['triplet'] = df.index//3
df_triplet_means = df.groupby('triplet').mean()
df_triplet_max = df.groupby('triplet').max()

df.head()

# # Active learning with uniform sampling

def choose_base_inds_numpy(y: np.ndarray, num_choose: int, how: Literal['max','min'] = 'max', strategy: Literal['uniform','skewed'] = 'uniform'):
    ind = np.argsort(y)
    N = y.shape[0]
    i = np.arange(N)
    if strategy=='skewed':
        if how=='max':
            p = (i - i.max())**2
        elif how=='min':
            p = i**2
    elif strategy=='uniform':
        p = np.ones_like(i)
    else: 
        raise ValueError
    p = p/p.sum()
    return np.random.choice(ind, size=num_choose, replace=False, p=p)


NUM_RUNS = 100
col_n = 'FE (Eth)'
col_i = 0

# # Ethylene maximization

# ## MLP ensemble

gde = GDEOptimizer(model_name="GP+Ph", acquisition="EI", quantity="FE (Eth)", maximize=True, output_dir="./out")

for d in range(NUM_RUNS):
    DNAME = Path(f'./MLP_F/MLP_F{d}')
    DNAME.mkdir(exist_ok=True, parents=True)
    df.to_csv(DNAME/'df.csv')
    chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3)

    expected_improvements = [None]*len(chosen_triplets)
    while len(chosen_triplets)<df_triplet_means.shape[0]:
        withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

        chosen_df = df[df['triplet'].isin(chosen_triplets)]

        best_ei, best_i = gde.step_within_data(chosen_triplets, withheld_triplets)
        
        expected_improvements.append(best_ei)
        maxtrip = withheld_triplets.index[best_i]
        chosen_triplets = np.append(chosen_triplets, maxtrip)
    
    print('')
    pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


# # ## Ph ensemble

# # In[ ]:


# ds = range(NUM_RUNS)
# for d in ds:
#     DNAME = Path(f'./Ph_F/Ph_F{d}')
#     DNAME.mkdir(exist_ok=True, parents=True)
#     df.to_csv(DNAME/'df.csv')
#     chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

#     i = 0
#     i_to_max = None
#     expected_improvements = [None]*len(chosen_triplets)
#     while len(chosen_triplets)<df_triplet_means.shape[0]:
#         withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

#         chosen_df = df[df['triplet'].isin(chosen_triplets)]
#         X, y, means, stds, _ = normalize_df_torch(chosen_df)

#         if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
#             i_to_max = i
#         print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

#         model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233)
#         try:
#             stats, predict = train_model_ens(X, y, model, num_iter=101, DNAME=DNAME, i=i)
#         except:
#             print('')
#             break

#         X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
#         mu, std = predict(X_test)
#         res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
#         ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min()[0]), minimize=False)
#         maxind = ei.argmax().item()
#         expected_improvements.append(ei.max().item())
#         maxtrip = test_df.index[maxind]
#         chosen_triplets = np.append(chosen_triplets, maxtrip)

#         i += 1
    
#     print('')
#     pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


# # ## GP

# # In[ ]:


# ds = range(NUM_RUNS)
# for d in ds:
#     DNAME = Path(f'./GP_F/GP_F{d}')
#     DNAME.mkdir(exist_ok=True, parents=True)
#     df.to_csv(DNAME/'df.csv')
#     chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

#     i = 0
#     i_to_max = None
#     expected_improvements = [None]*len(chosen_triplets)
#     while len(chosen_triplets)<df_triplet_means.shape[0]:
#         withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

#         chosen_df = df[df['triplet'].isin(chosen_triplets)]
#         X, y, means, stds, _ = normalize_df_torch(chosen_df)

#         if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
#             i_to_max = i
#         print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

#         try:
#             stats, predict = train_GP_model(X, y, num_iter=101, DNAME=DNAME, i=i)
#         except:
#             print('')
#             break

#         X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
#         mu, std = predict(X_test)
#         res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
#         ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min()[0]), minimize=False)
#         maxind = ei.argmax().item()
#         expected_improvements.append(ei.max().item())
#         maxtrip = test_df.index[maxind]
#         chosen_triplets = np.append(chosen_triplets, maxtrip)

#         i += 1
    
#     print('')
#     pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


# # ## GP+Ph

# # In[ ]:


# ds = range(NUM_RUNS)
# for d in ds:
#     DNAME = Path(f'./GP_Ph_F/GP_Ph_F{d}')
#     DNAME.mkdir(exist_ok=True, parents=True)
#     df.to_csv(DNAME/'df.csv')
#     chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

#     i = 0
#     i_to_max = None
#     expected_improvements = [None]*len(chosen_triplets)
#     while len(chosen_triplets)<df_triplet_means.shape[0]:
#         withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

#         chosen_df = df[df['triplet'].isin(chosen_triplets)]
#         X, y, means, stds, _ = normalize_df_torch(chosen_df)

#         if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
#             i_to_max = i
#             print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20)
#             break
#         print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

#         model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233) 
#         try:
#             stats, predict = train_GP_Ph_model(X, y, model, num_iter=101, DNAME=DNAME, i=i, plot=False)
#         except:
#             print('')
#             break

#         X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
#         mu, std = predict(X_test)
#         res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
#         ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min()[0]), minimize=False)
#         maxind = ei.argmax().item()
#         expected_improvements.append(ei.max().item())
#         maxtrip = test_df.index[maxind]
#         chosen_triplets = np.append(chosen_triplets, maxtrip)

#         i += 1
#     print('')
#     pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


# # # Post-process

# # In[17]:


# def process_runs_mean(dname):
#     all_df = []
#     for p in dname.iterdir():
#         if not p.is_dir(): continue
#         df = pd.read_csv(p/'df.csv', index_col=0)
#         chosen_triplets = pd.read_csv(p/'chosen_triplets.csv', index_col=0)
#         df_triplets_mean = df.groupby('triplet').mean()
#         chosen_triplets.loc[:, 'cummax FE'] = df_triplets_mean.loc[chosen_triplets['chosen_triplets'], 'FE (Eth)'].cummax().values

#         i0 = 2
#         chosen_triplets['step'] = chosen_triplets.index - i0
#         chosen_triplets['dname'] = p.stem
#         all_df.append(chosen_triplets)
#     all_df = pd.concat(all_df, axis=0)
#     return all_df


# # In[ ]:


# for dname in ['MLP_F', 'Ph_F', 'GP_F', 'GP_Ph_F']:
#     steps_to_finish = []
#     for p in Path(dname).iterdir():
#         if not p.is_dir(): continue
#         _df = pd.read_csv(p/'df.csv', index_col=0)
#         chosen_triplets = pd.read_csv(p/'chosen_triplets.csv', index_col=0)
#         df_triplets_mean = _df.groupby('triplet').mean()
#         chosen_triplets.loc[:, 'cummax FE'] = df_triplets_mean.loc[chosen_triplets['chosen_triplets'], 'FE (Eth)'].cummax().values

#         i0 = 2
#         chosen_triplets['step'] = chosen_triplets.index - i0
#         chosen_triplets['dname'] = p.stem
#         steps_to_finish.append(chosen_triplets.loc[chosen_triplets['cummax FE']>0.245, 'step'].min())
#     sf = np.array(steps_to_finish)
#     sf[sf<0] = 0
#     mean = np.mean(sf[~np.isnan(sf)])
#     std = np.std(sf[~np.isnan(sf)])
#     af = 13 / mean
#     print(dname, mean, std, af)


# # In[ ]:


# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,8), sharex=True, sharey=True)
# all_df = []
# for i, dname in enumerate(['MLP_F', 'Ph_F', 'GP_F', 'GP_Ph_F']):
#     _df = process_runs_mean(Path(dname))
#     _df = _df[_df['step']>=0]
#     sns.lineplot(data=_df, x='step', y='cummax FE', hue='dname', legend=False, ax=ax[i//2, i%2])
#     ax[i//2, i%2].set_title(dname)
#     ax[i//2, i%2].set_ylabel('min(FE)')
#     all_df.append(_df)
# all_df = pd.concat(all_df, axis=0)
# fig.tight_layout()
# plt.show()


# # In[32]:


# all_df['Model'] = all_df['dname'].map(lambda x: '_'.join(x.split('_')[:-1]))
# plt.figure(figsize=(3.75,3))
# sns.lineplot(data=all_df, x='step', y='cummax FE', hue='Model', marker='o', ms=5)
# plt.ylabel('max FE (Eth)')
# plt.xlabel('Step')
# plt.savefig('./active-learning.pdf', bbox_inches='tight', pad_inches=0.1)
# plt.show()

