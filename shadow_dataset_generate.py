def create_shadow_data(k):
    data = pd.read_csv(r'./data/census-income-new.csv')

    columns = ['age', 'class of worker', 'detailed industry recode', 'detailed occupation recode', 'education',
               'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code',
               'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union',
               'reason for unemployment', 'full or part time employment stat', 'capital gains',
               'capital losses', 'dividends from stocks', 'tax filer stat', 'region of previous residence',
               'state of previous residence', 'detailed household and family stat',
               'detailed household summary in household',
               'instance weight',
               'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg',
               'live in this house 1 year ago', 'migration prev res in sunbelt',
               'num persons worked for employer', 'family members under 18', 'country of birth father',
               'country of birth mother', 'country of birth self', 'citizenship', 'own business or self employed',
               'fill inc questionnaire for veterans admin', 'veterans benefits', 'weeks worked in year',
               'year', 'income']
    # print(len(columns)) # 41+1 = 42

    age = 0 - 90
    class_of_worker = [' Not in universe', ' Federal government', ' Local government', ' Never worked', 'Private',
                       ' Self-employed-incorporated', ' Self-employed-not incorporated',
                       ' State government, Without pay']
    detailed_industry_recode = [0, 40, 44, 2, 43, 47, 48, 1, 11, 19, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 4, 42, 45,
                                5, 15, 16, 22, 29, 31, 50, 14, 17, 18, 28, 3, 30, 41, 46, 51, 12, 13, 21, 23, 26, 6, 7,
                                9, 49, 27, 8, 10, 20]
    detailed_occupation_recode = [0, 12, 31, 44, 19, 32, 10, 23, 26, 28, 29, 42, 40, 34, 14, 36, 38, 2, 20, 25, 37, 41,
                                  27, 24, 30, 43, 33, 16, 45, 17, 35, 22, 18, 39, 3, 15, 13, 46, 8, 21, 9, 4, 6, 5, 1,
                                  11, 7]
    education = [' Children', ' 7th and 8th grade', ' 9th grade', ' 10th grade', ' High school graduate', ' 11th grade',
                 ' 12th grade no diploma', ' 5th or 6th grade', ' Less than 1st grade', ' Bachelors degree(BA AB BS)',
                 ' 1st 2nd 3rd or 4th grade', ' Some college but no degree', ' Masters degree(MA MS MEng MEd MSW MBA)',
                 ' Associates degree-occup /vocational', ' Associates degree-academic program',
                 ' Doctorate degree(PhD EdD)', ' Prof school degree (MD DDS DVM LLB JD)']
    wage_per_hour = []
    enroll_in_edu_inst_last_wk = [' Not in universe', ' High school', ' College or university']
    marital_stat = [' Never married', ' Married-civilian spouse present', ' Married-spouse absent', ' Separated',
                    ' Divorced', ' Widowed', ' Married-A F spouse present']
    major_industry_code = [' Not in universe or children', ' Entertainment', ' Social services', ' Agriculture',
                           ' Education', ' Public administration', ' Manufacturing-durable goods',
                           ' Manufacturing-nondurable goods', ' Wholesale trade', ' Retail trade',
                           ' Finance insurance and real estate', ' Private household services',
                           ' Business and repair services', ' Personal services except private HH', ' Construction',
                           ' Medical except hospital', ' Other professional services', ' Transportation',
                           ' Utilities and sanitary services', ' Mining', ' Communications', ' Hospital services',
                           ' Forestry and fisheries', ' Armed Forces']
    major_occupation_code = [' Not in universe', ' Professional specialty', ' Other service',
                             ' Farming forestry and fishing', ' Sales', ' Adm support including clerical',
                             ' Protective services', ' Handlers equip cleaners etc ',
                             ' Precision production craft & repair', ' Technicians and related support',
                             ' Machine operators assmblrs & inspctrs', ' Transportation and material moving',
                             ' Executive admin and managerial', ' Private household services', ' Armed Forces']
    race = [' White', ' Black', ' Other', ' Amer Indian Aleut or Eskimo', ' Asian or Pacific Islander']
    hispanic_origin = [' Mexican (Mexicano)', ' Mexican-American', ' Puerto Rican', ' Central or South American',
                       ' All other', ' Other Spanish', ' Chicano', ' Cuban', ' Do not know', ' NA']
    sex = [' Female', ' Male']
    member_of_a_labor_union = [' Not in universe', ' No', ' Yes']
    reason_for_unemployment = [' Not in universe', ' Re-entrant', ' Job loser - on layoff', ' New entrant',
                               ' Job leaver', ' Other job loser']
    full_or_part_time_employment_stat = [' Children or Armed Forces', ' Full-time schedules', ' Unemployed part- time',
                                         ' Not in labor force', ' Unemployed full-time',
                                         ' PT for non-econ reasons usually FT', ' PT for econ reasons usually PT',
                                         ' PT for econ reasons usually FT']
    capital_gains = []
    capital_losses = []
    dividends_from_stocks = []
    tax_filer_stat = [' Nonfiler', ' Joint one under 65 & one 65+', ' Joint both under 65', ' Single',
                      ' Head of household', ' Joint both 65+']
    region_of_previous_residence = [' Not in universe', ' South', ' Northeast', ' West', ' Midwest', ' Abroad']
    state_of_previous_residence = [' Not in universe', ' Utah', ' Michigan', ' North Carolina', ' North Dakota',
                                   ' Virginia', ' Vermont', ' Wyoming', ' West Virginia', ' Pennsylvania', ' Abroad',
                                   ' Oregon', ' California', ' Iowa', ' Florida', ' Arkansas', ' Texas',
                                   ' South Carolina', ' Arizona', ' Indiana', ' Tennessee', ' Maine', ' Alaska',
                                   ' Ohio', ' Montana', ' Nebraska', ' Mississippi', ' District of Columbia',
                                   ' Minnesota', ' Illinois', ' Kentucky', ' Delaware', ' Colorado', ' Maryland',
                                   ' Wisconsin', ' New Hampshire', ' Nevada', ' New York', ' Georgia', ' Oklahoma',
                                   ' New Mexico', ' South Dakota', ' Missouri', ' Kansas', ' Connecticut', ' Louisiana',
                                   ' Alabama', ' Massachusetts', ' Idaho', ' New Jersey']
    detailed_household_and_family_stat = [' Child <18 never marr not in subfamily',
                                          ' Other Rel <18 never marr child of subfamily RP',
                                          ' Other Rel <18 never marr not in subfamily',
                                          ' Grandchild <18 never marr child of subfamily RP',
                                          ' Grandchild <18 never marr not in subfamily', ' Secondary individual',
                                          ' In group quarters', ' Child under 18 of RP of unrel subfamily',
                                          ' RP of unrelated subfamily', ' Spouse of householder', ' Householder',
                                          ' Other Rel <18 never married RP of subfamily',
                                          ' Grandchild <18 never marr RP of subfamily',
                                          ' Child <18 never marr RP of subfamily',
                                          ' Child <18 ever marr not in subfamily',
                                          ' Other Rel <18 ever marr RP of subfamily',
                                          ' Child <18 ever marr RP of subfamily', ' Nonfamily householder',
                                          ' Child <18 spouse of subfamily RP', ' Other Rel <18 spouse of subfamily RP',
                                          ' Other Rel <18 ever marr not in subfamily',
                                          ' Grandchild <18 ever marr not in subfamily',
                                          ' Child 18+ never marr Not in a subfamily',
                                          ' Grandchild 18+ never marr not in subfamily',
                                          ' Child 18+ ever marr RP of subfamily',
                                          ' Other Rel 18+ never marr not in subfamily',
                                          ' Child 18+ never marr RP of subfamily',
                                          ' Other Rel 18+ ever marr RP of subfamily',
                                          ' Other Rel 18+ never marr RP of subfamily',
                                          ' Other Rel 18+ spouse of subfamily RP',
                                          ' Other Rel 18+ ever marr not in subfamily',
                                          ' Child 18+ ever marr Not in a subfamily',
                                          ' Grandchild 18+ ever marr not in subfamily',
                                          ' Child 18+ spouse of subfamily RP', ' Spouse of RP of unrelated subfamily',
                                          ' Grandchild 18+ ever marr RP of subfamily',
                                          ' Grandchild 18+ never marr RP of subfamily',
                                          ' Grandchild 18+ spouse of subfamily RP']
    detailed_household_summary_in_household = [' Child under 18 never married', ' Other relative of householder',
                                               ' Nonrelative of householder', ' Spouse of householder', ' Householder',
                                               ' Child under 18 ever married', ' Group Quarters- Secondary individual',
                                               ' Child 18 or older']
    instance_weight = []
    migration_code_change_in_msa = [' Not in universe', ' Nonmover', ' MSA to MSA', ' NonMSA to nonMSA',
                                    ' MSA to nonMSA', ' NonMSA to MSA', ' Abroad to MSA', ' Not identifiable',
                                    ' Abroad to nonMSA']
    migration_code_change_in_reg = [' Not in universe', ' Nonmover', ' Same county', ' Different county same state',
                                    ' Different state same division', ' Abroad, Different region',
                                    ' Different division same region']
    migration_code_move_within_reg = [' Not in universe', ' Nonmover', ' Same county', ' Different county same state',
                                      ' Different state in West', ' Abroad', ' Different state in Midwest',
                                      ' Different state in South', ' Different state in Northeast']
    live_in_this_house_1_year_ago = [' Not in universe under 1 year old', ' Yes', ' No']
    migration_prev_res_in_sunbelt = [' Not in universe', ' Yes', ' No']
    num_persons_worked_for_employer = 0 - 6
    family_members_under_18 = [' Both parents present', ' Neither parent present', ' Mother only present',
                               ' Father only present', ' Not in universe']
    country_of_birth_father = [' Mexico', ' United-States', ' Puerto-Rico', ' Dominican-Republic', ' Jamaica', ' Cuba',
                               ' Portugal', ' Nicaragua', ' Peru', ' Ecuador', ' Guatemala', ' Philippines', ' Canada',
                               ' Columbia', ' El-Salvador', ' Japan', ' England', ' Trinadad&Tobago', ' Honduras',
                               ' Germany', ' Taiwan', ' Outlying-U S (Guam USVI etc)', ' India', ' Vietnam', ' China',
                               ' Hong Kong', ' Cambodia', ' France', ' Laos', ' Haiti', ' South Korea', ' Iran',
                               ' Greece', ' Italy', ' Poland', ' Thailand', ' Yugoslavia', ' Holand-Netherlands',
                               ' Ireland', ' Scotland', ' Hungary', ' Panama']
    country_of_birth_mother = [' India', ' Mexico', ' United-States', ' Puerto-Rico', ' Dominican-Republic', ' England',
                               ' Honduras', ' Peru', ' Guatemala', ' Columbia', ' El-Salvador', ' Philippines',
                               ' France', ' Ecuador', ' Nicaragua', ' Cuba', ' Outlying-U S (Guam USVI etc)',
                               ' Jamaica', ' South Korea', ' China', ' Germany', ' Yugoslavia', ' Canada', ' Vietnam',
                               ' Japan', ' Cambodia', ' Ireland', ' Laos', ' Haiti', ' Portugal', ' Taiwan',
                               ' Holand-Netherlands', ' Greece', ' Italy', ' Poland', ' Thailand', ' Trinadad&Tobago',
                               ' Hungary', ' Panama', ' Hong Kong', ' Scotland', ' Iran']
    country_of_birth_self = [' United-States', ' Mexico', ' Puerto-Rico', ' Peru', ' Canada', ' South Korea', ' India',
                             ' Japan', ' Haiti', ' El-Salvador', ' Dominican-Republic', ' Portugal', ' Columbia',
                             ' England', ' Thailand', ' Cuba', ' Laos', ' Panama', ' China', ' Germany', ' Vietnam',
                             ' Italy', ' Honduras', ' Outlying-U S (Guam USVI etc)', ' Hungary', ' Philippines',
                             ' Poland', ' Ecuador', ' Iran', ' Guatemala', ' Holand-Netherlands', ' Taiwan',
                             ' Nicaragua', ' France', ' Jamaica', ' Scotland', ' Yugoslavia', ' Hong Kong',
                             ' Trinadad&Tobago', ' Greece', ' Cambodia', ' Ireland']
    citizenship = [' Native- Born in the United States', ' Foreign born- Not a citizen of U S ',
                   ' Native- Born in Puerto Rico or U S Outlying', ' Native- Born abroad of American Parent(s)',
                   ' Foreign born- U S citizen by naturalization']
    own_business_or_self_employed = [0, 2, 1]
    fill_inc_questionnaire_for_veteran_admin = [' Not in universe', ' Yes', ' No']
    veterans_benefits = [0, 2, 1]
    weeks_worked_in_year = 0 - 52
    year = [94, 95]

    income = ['-50000', ' 50000+.']

    datasets = []

    for i in range(k):
        datasets.append([random.randint(0, 90), random.choice(class_of_worker), random.choice(detailed_industry_recode),
                         random.choice(detailed_occupation_recode), random.choice(education),
                         data["wage per hour"][random.randint(1, 23990)], random.choice(enroll_in_edu_inst_last_wk),
                         random.choice(marital_stat), random.choice(major_industry_code),
                         random.choice(major_occupation_code), random.choice(race), random.choice(hispanic_origin),
                         random.choice(sex), random.choice(member_of_a_labor_union),
                         random.choice(reason_for_unemployment), random.choice(full_or_part_time_employment_stat),
                         data["capital gains"][random.randint(1, 23990)],
                         data["capital losses"][random.randint(0, 23990)],
                         data["dividends from stocks"][random.randint(1, 23990)], random.choice(tax_filer_stat),
                         random.choice(region_of_previous_residence), random.choice(state_of_previous_residence),
                         random.choice(detailed_household_and_family_stat),
                         random.choice(detailed_household_summary_in_household),
                         data["instance weight"][random.randint(1, 23990)],
                         random.choice(migration_code_change_in_msa), random.choice(migration_code_change_in_reg),
                         random.choice(migration_code_move_within_reg),
                         random.choice(live_in_this_house_1_year_ago), random.choice(migration_prev_res_in_sunbelt),
                         random.randint(0, 6), random.choice(family_members_under_18),
                         random.choice(country_of_birth_father),
                         random.choice(country_of_birth_mother), random.choice(country_of_birth_self),
                         random.choice(citizenship), random.choice(own_business_or_self_employed),
                         random.choice(fill_inc_questionnaire_for_veteran_admin),
                         random.choice(veterans_benefits), random.randint(0, 52), random.choice(year),
                         random.choice(income)])
    df = pd.DataFrame(datasets, columns=columns)
    return df


def datafilter(data, model):
    with torch.no_grad():
        model.eval()
        # # print(model.fc4.weight[0][0])
        data_A = data[data[FEATRUE] == TAG_A]  # Separate dataframe into both classes
        data_B = data[data[FEATRUE] == TAG_B]
        # print(data_A.shape)
        # print(data_B.shape)
        # print("Shadow-datasets: {} + {} = {}".format(data_A.shape, data_B.shape, data_A.shape + data_B.shape))

        # data[FEATRUE] = " ?"

        # NORMALIZE CONTINUOUS FEATURES
        # Make a list of all continous features
        cont_feats = ['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks',
                      'instance weight',
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
        data_income = data_income.values  # Turn into a numpy array
        data_categorical.drop('income', axis=1, inplace=True)  # get rid of 'income' column from dataframe
        processed_data = np.concatenate((data_cont, data_categorical), axis=1)
        # print(data_income.sum())

        x_CompositeDatas = torch.Tensor(processed_data)
        # y_CompositeDatas = torch.Tensor(data_income)

        prediction = model(x_CompositeDatas)
        pred = prediction.data.max(1, keepdim=True)[1]
        # print(pred.sum())
    return torch.as_tensor(x_CompositeDatas.numpy(), dtype=torch.float32), torch.as_tensor(pred.squeeze(),
                                                                                           dtype=torch.int64)
