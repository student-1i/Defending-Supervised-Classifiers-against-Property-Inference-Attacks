def dataClean(data, is_TrainSet=False, x=0, y=0):
    data_A = data[data[FEATRUE] == TAG_A]  # Separate dataframe into both classes
    data_B = data[data[FEATRUE] == TAG_B]
    if is_TrainSet == True:
        data_A = data_A.sample(n=x)  # sample to get same amount, no set random_state  is random,10000
        data_B = data_B.sample(n=y)
        print("Trainset number:{} + {} = {}".format(x, y, x + y))
    data = pd.concat([data_A, data_B], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)

    # NORMALIZE CONTINUOUS FEATURES
    # Make a list of all continous features
    cont_feats = ['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks', 'instance weight',
                  'num persons worked for employer', 'weeks worked in year']
    cat_feats = ['class of worker', 'detailed industry recode', 'detailed occupation recode', 'education',
                 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code',
                 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment',
                 'full or part time employment stat', 'tax filer stat', 'region of previous residence',
                 'state of previous residence', 'detailed household and family stat',
                 'detailed household summary in household', 'migration code-change in msa',
                 'migration code-change in reg',
                 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt',
                 'family members under 18', 'country of birth father', 'country of birth mother',
                 'country of birth self', 'citizenship', 'own business or self employed',
                 'fill inc questionnaire for veterans admin', 'veterans benefits', 'year', 'income']
    data_cont = data[cont_feats]  # 2 new dataframes for continuous and categorical features
    data_categorical = data[cat_feats]
    normalized_data = data_cont.copy()

    for feature in cont_feats:
        mean = data_cont[feature].mean()
        std = data_cont[feature].std()
        normalized_data[feature] = (data_cont[feature] - mean) / std

    data_cont = normalized_data
    # data_cont.drop('instance weight', axis=1, inplace=True)  # get rid of 'instance weight' column from dataframe
    data_cont = data_cont.values  # Turn it into a numpy array

    # ENCODE CATEGORICAL FEATURES
    label_encoder = LabelEncoder()
    encoded_data = data_categorical.copy()

    for feature in cat_feats:
        label_encoder.fit(data_categorical[feature])
        encoded_data[feature] = label_encoder.transform(data_categorical[feature])
    data_categorical = encoded_data
    data_income = data_categorical["income"]
    data_income = data_income.values
    data_categorical.drop('income', axis=1, inplace=True)  # get rid of 'income' column from dataframe
    processed_data = np.concatenate((data_cont, data_categorical), axis=1)

    return torch.tensor(processed_data, dtype=torch.float32), torch.tensor(data_income, dtype=torch.int64)
