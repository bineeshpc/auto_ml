""" 
MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)

"""
import pandas as pd
import numpy as np
import os
import logger
import utils
from exploratory_data_analysis import detect_outliers
import config_parser
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from transformer import Transformer

filename = '/tmp/arguments.pkl'
args = utils.load(filename)

def extract_title(x):
    d = dict(Mlle='Miss',
            Ms='Miss',
            Mme='Mrs')
    for i in ['Lady', 'the Countess','Countess','Capt',
     'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
     d[i] = 'Rare'

    a = x.split(',')[1].split('.')[0].strip()
    if a in d.keys():
        a = d[a]
    return a

def combine_df_same_length(df1, df2):
    length = len(df2)
    d = dict([(i, [])for i in df1.columns])
    for column_ in df2.columns:
        d[column_] = []
    for i in range(length):
        for column_ in df1.columns:
            d[column_].append(df1[column_][i])
        for column_ in df2.columns:
            d[column_].append(df2[column_][i])
    df3 = pd.DataFrame(d)
    return df3
    

def recreate_df(df3, good_values_df, filled_values_df, c_id):
    x = df3[c_id] == df3.index + 1
    df4 = df3.set_index(c_id)
    df5 = good_values_df.set_index(c_id)
    df6 = filled_values_df.set_index(c_id)
    df7 = pd.concat([df5['Age'], df5['Age']])
    #df8 = df3['Age'] = df7
    df7.index = range(len(df7))
    # print(df7.index)
    # print(df3.index)
    df3['Age'] = df7
    return df3
    
                
               

def replace_nan_transformer(df, column, value):
    """ Replace nan of column with value
    """
    df1 = df.copy()
    df1[column] = df[column].fillna(value)
    return df1


def label_encoder_transformer(df, column):
    """ Encode the column of df with label encoder
    """
    df1 = df.copy()
    model = LabelEncoder()
    result = model.fit_transform(df[column])
    df1[column] = result
    return df1


def one_hot_encoder_transformer(df, column):
    """ Encode the column of df with label encoder
    """
    df1 = df.copy()
    df1 = pd.get_dummies(df1, columns=[column])
    return df1

def function_apply_transformer(df, function, column, new_column):
    """ Apply function to df on column delete the column and 
    the result will be named as a new_column 
    """
    df1 = df.copy()
    extracted = df[column].apply(function)
    df1.drop(column, axis='columns')
    df1[new_column] = extracted
    return df1

def drop_columns(df, columns):
    return df.drop(columns, axis='columns')
    


def fill_column(df, text_df, column, scaler, regressor):
    """ column of df is predicted using text_df and a regressor
    """
    df1 = df.copy()
    df3 = combine_df_same_length(df, text_df)
    null_values_df = df3[df3[column].isnull()]
    good_values_df = df3[df3[column].isnull() != True]
    # print(null_values_df.shape)
    # print(good_values_df.shape)
    # print(df3.shape)
    # print(good_values_df.columns, column)
    
    X_train = good_values_df.drop(column, axis='columns').values
    y_train = good_values_df[column].values.reshape(-1, 1)
    # print(X_train.shape)
    # print(y_train.shape)
    steps = [('scaler', scaler),
                ('regressor', regressor)
            ]
    pipeline = Pipeline(steps)
    model = pipeline.fit(X_train, y_train)
    X_test = null_values_df.drop(column, axis='columns').values
    y_pred = model.predict(X_test)

    filled_values_df = null_values_df.copy()
    filled_values_df[column] = y_pred
    df4 = recreate_df(df3, good_values_df, filled_values_df, 'PassengerId')
    df1[column] = df4[column]
    return df1
    
def ticket_modifier(x):
    y = x.split()
    if len(y) > 1:
        return y[0].replace('/', '').replace('.', '').strip()
    else:
        return 'X'

def create_family_size(df, column_formed, column_1, column_2):
    df[column_formed] = df[column_1] + df[column_2] + 1
    return df

def create_family_features(dataset):
    dataset['Single'] = dataset['Family_Size'].map(lambda s: 1 if s == 1 else 0)
    dataset['Small_Family'] = dataset['Family_Size'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Medium_Family'] = dataset['Family_Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['Large_Family'] = dataset['Family_Size'].map(lambda s: 1 if s >= 5 else 0)
    return dataset

def create_age_features(dataset):
    dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 35), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 50), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 65), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 65, 'Age'] = 5
    dataset['Age_Categories'] = dataset['Age'].astype('int')
    return dataset


def drop_outliers(df, outliers):
    """ Remove outliers given in iterable outliers
    """
    try:
        df = df.drop(outliers, axis = 0).reset_index(drop=True)
    except KeyError:
        pass
        # transformers_logger.info('{} not found in df'.format(outliers)
    return df


def transform_bool(type_, df):
    if type_ == 'bool_':
        transformer = Transformer()
        # df = transformer.do_transformation('identity transformer', (lambda x: x), (df,), {})
    return df

def transform_categorical(type_, df):
    if type_ == 'categorical':
        transformer = Transformer()
        # df = transformer.do_transformation('replace nan of embarked column', replace_nan_transformer, 
        # (df, 'Embarked', df['Embarked'].mode()[0]), {})
        # for column_ in df.columns:
        #     df = transformer.do_transformation('one hot encoder transformer', one_hot_encoder_transformer, 
        # (df, column_), {})
    return df

def transform_text(type_, df):

    if type_ == 'text':
        transformer = Transformer()
        # df = transformer.do_transformation('extract title transformer', function_apply_transformer,
        #     (df, extract_title, 'Name', 'Title'),
        #     {})
        # df = transformer.do_transformation('one hot encoder transformer', one_hot_encoder_transformer, 
        # (df, 'Title'), {})
        
        # df = transformer.do_transformation('Extract ticket info', 
        # function_apply_transformer, (df, ticket_modifier, 'Ticket', 'Ticket_Modified'), {})
        
        # # dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
        # df = transformer.do_transformation('replace nan of cabin column', replace_nan_transformer, 
        # (df, 'Cabin', 'X'), {})

        # df = transformer.do_transformation('Extract Cabin info first letter', 
        # function_apply_transformer, (df, (lambda x: x[0]), 'Cabin', 'Cabin_Modified'), {})
        
        # df = transformer.do_transformation('label encoder transformer', label_encoder_transformer, 
        # (df, 'Cabin_Modified'), {})
        # df = transformer.do_transformation('label encoder transformer', label_encoder_transformer, 
        # (df, 'Ticket_Modified'), {})
        

        df = transformer.do_transformation('drop columns', drop_columns,
        (df, df.columns),
        {})
    
    return df

def transform_numerical(type_, df):
    if type_ == 'numerical':
        transformer = Transformer()
        # directory = config_parser.get_configuration(args.configfile).get_directory('transformer')
        # # transformers_logger.info(directory)
        # df = transformer.do_transformation('replace nan of fare column', replace_nan_transformer, 
        # (df, 'Fare', df['Fare'].median()), {})
        # # need to use the generated column title to fill/complete the missing entries in age
        # text_df = pd.read_csv(os.path.join(directory, 'text.csv'))
        
        # df = transformer.do_transformation('fill missing entries in age', fill_column, 
        # (df, text_df, 'Age', StandardScaler(), LinearRegression()),
        # {}
        # )
        # df = transformer.do_transformation('make age group into categories',
        # create_age_features,
        # (df,),
        # {}
        # ) 
        # df = transformer.do_transformation('one hot encoder transformer',
        #     one_hot_encoder_transformer, 
        # (df, 'Age_Categories'),
        #     {})
        # df = transformer.do_transformation('Create family size feature', create_family_size, 
        # (df, 'Family_Size', 'Parch', 'SibSp'),
        # {}
        # )
        # df = transformer.do_transformation('Create family size special features',
        #     create_family_features, 
        # (df, ),
        # {}
        # )               

        df = transformer.do_transformation('drop columns', drop_columns,
        (df, ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']),
        {})
    return df
