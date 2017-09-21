import numpy   as np
import pandas  as pd

import remap_values as rv


from sklearn.decomposition import PCA
from sklearn               import linear_model

import pickle



# Fill null values in tax info

def regression_taxamount( inp_df ):

    # get values, and log them
    foo = np.log10( inp_df[['taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','structuretaxvaluedollarcnt']] ).copy()
    foo['parcelid'] = inp_df['parcelid']
    
    # Bar is just not null values, for regression
    bar = foo.dropna()
    
        
    # Perform regressions
    land_reg = linear_model.LinearRegression()
    tax_reg  = linear_model.LinearRegression()
    amt_reg  = linear_model.LinearRegression()
    str_reg  = linear_model.LinearRegression()

    
    land_reg.fit( bar['taxamount'        ].to_frame(), bar['landtaxvaluedollarcnt'].to_frame() )
    tax_reg.fit ( bar['taxamount'        ].to_frame(), bar[    'taxvaluedollarcnt'].to_frame() )
    amt_reg.fit ( bar['taxvaluedollarcnt'].to_frame(), bar[            'taxamount'].to_frame() )
    str_reg.fit ( bar['taxvaluedollarcnt'].to_frame(), bar['structuretaxvaluedollarcnt'].to_frame() )

    # Get the id numbers for the rows with NaN values
    for id_num in foo[ foo.drop( 'structuretaxvaluedollarcnt',axis=1).isnull().any(axis=1) ]['parcelid'].values:
        
        # Find which row it is
        row_num = foo.loc[ foo['parcelid'] == id_num ].index
        
        # Depending on what's missing, fill it in
        
        
        if( foo.ix[ row_num, 'landtaxvaluedollarcnt' ].isnull().values ):
            foo.ix[ row_num, 'landtaxvaluedollarcnt' ] = land_reg.predict( foo.ix[ row_num, 'taxamount'         ].values.reshape(-1,1) )
            
        if( foo.ix[ row_num,     'taxvaluedollarcnt' ].isnull().values ):
            foo.ix[ row_num,     'taxvaluedollarcnt' ] =  tax_reg.predict( foo.ix[ row_num, 'taxamount'         ].values.reshape(-1,1) )
            
        if( foo.ix[ row_num,             'taxamount' ].isnull().values ):
            foo.ix[ row_num,             'taxamount' ] =  amt_reg.predict( foo.ix[ row_num, 'taxvaluedollarcnt' ].values.reshape(-1,1) )

    for id_num in foo[ foo.isnull().any(axis=1) ]['parcelid'].values:

        row_num = foo.loc[ foo['parcelid'] == id_num ].index
        
        if( foo.ix[ row_num,'structuretaxvaluedollarcnt'].isnull().values ):
            foo.ix[ row_num,'structuretaxvaluedollarcnt'] =  tax_reg.predict( foo.ix[ row_num,'taxvaluedollarcnt'].values.reshape(-1,1) )

            
    return foo[['taxamount','taxvaluedollarcnt','landtaxvaluedollarcnt','structuretaxvaluedollarcnt']]



# Fill in nulls for number of bathrooms

def reg_num_bath( inp_df ):
    
    foo = inp_df[['calculatedbathnbr','structuretaxvaluedollarcnt','parcelid']].copy()
    foo['structuretaxvaluedollarcnt'] = np.log10( foo['structuretaxvaluedollarcnt'] ) 
    foo['calculatedbathnbr'         ] = 2 *       foo['calculatedbathnbr'         ]
    
    bar = foo.dropna()
        
    # Perform regressions
    bath_reg = linear_model.LinearRegression()
    
    bath_reg.fit( bar['structuretaxvaluedollarcnt'].to_frame(), bar['calculatedbathnbr'].to_frame() )
    
    # Don't need to worry about going through rows
    # Locate bad indexes, and fill those spots with predicted values
    null_index = foo.isnull().any(axis=1)

    # Do regression
    foo.ix[ null_index, ['calculatedbathnbr'] ] = \
        bath_reg.predict( foo.ix[ null_index, 'structuretaxvaluedollarcnt' ].values.reshape(-1,1) )
        
    # Convert to 1, 1.5 2 ... baths
    foo        ['calculatedbathnbr'] = foo['calculatedbathnbr'].round()/2.
    
    # Lowst valid is 1
    foo.ix[ foo['calculatedbathnbr'] < 1,  'calculatedbathnbr'] = 1
    
    # Regression only valid for up to 3, VAST majority in this range
    foo.ix[ null_index & (foo.loc[ null_index ]['calculatedbathnbr'] > 3), 'calculatedbathnbr' ] = 3
    
    return foo['calculatedbathnbr']



def reg_unit_sqft( inp_df ):

    
    # Only things we need for regression
    foo = inp_df[['unit_sqft','n_bath','log_str_tax']].copy()
    
    foo['unit_sqft'] = np.log10( foo['unit_sqft'] +1 )
    
    # Bar is just not null values, for regression
    bar = foo.dropna()
    
    # Perform regressions
    unit_reg = linear_model.LinearRegression()
    unit_reg.fit( bar[['n_bath','log_str_tax']], bar['unit_sqft'].to_frame() )
    
    
    # Locate bad indexes, and fill those spots with predicted values
    null_index = foo.isnull().any(axis=1)


    # Do regression
    foo.ix[ null_index, ['unit_sqft'] ] = unit_reg.predict( foo.ix[ null_index, ['n_bath','log_str_tax'] ].values )#.reshape(-1,1) )

    return 10**foo['unit_sqft']



def reg_rooms( inp_df ):

    
    # Only things we need for regression
    foo = inp_df[['n_rooms','n_bath','unit_sqft','parcelid']].copy()
    
    foo['unit_sqft'] = np.log10( foo['unit_sqft'] )
    
    
    # Bar is just not null values, for regression
    bar = foo.loc[ foo['n_rooms']>0 ].dropna()
    
    
    # We will attempt to predict the difference between n_rooms, n_bath
    bar['diff'] = bar['n_rooms']-bar['n_bath']

    
    # Perform regressions
    room_reg = linear_model.LinearRegression()
    room_reg.fit( bar['unit_sqft'].to_frame(), bar['n_rooms'].to_frame() )    
    
    # Locate bad indexes, and fill those spots with predicted values
    null_index = foo.isnull().any(axis=1) | ( foo['n_rooms'] < 1 )
    
    
    # Do regression
    foo.ix[ null_index, ['n_rooms'] ] = room_reg.predict( foo.ix[ null_index, 'unit_sqft' ].values.reshape(-1,1) ).round()

    foo['diff'] = foo['n_rooms']-foo['n_bath']
        
    # if n_rooms-6.5 <= diff, n_rooms-6.5 <= n_rooms-n_bath
    foo.ix[ foo['n_rooms']-6.5 > foo['diff'], 'n_rooms' ] = ( foo.loc[ foo['n_rooms']-6.5 > foo['diff'] ]['n_rooms']-6.5 ).round()
    
    foo.ix[ foo['n_rooms']<0, 'n_rooms' ] = foo.ix[ foo['n_rooms']>0, 'n_rooms' ].min()
    
    return foo['n_rooms']



def build_data( my_df ):

    use_list = ['parcelid',
    'logerror','transactiondate','airconditioningtypeid','pooltypeid10', # Hotub
    'poolsizesum','garagetotalsqft','calculatedbathnbr','fireplacecnt',
    'finishedsquarefeet12','lotsizesquarefeet','taxdelinquencyflag','yearbuilt',
    'structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount',
    'decktypeid','numberofstories','buildingqualitytypeid','unitcnt','roomcnt' ]
    
    my_df = my_df[ use_list ].copy()
    
    my_df['transactiondate' ] =   pd.to_datetime  ( my_df['transactiondate'] )
    my_df['time_since_built'] = ( pd.DatetimeIndex( my_df['transactiondate'] ).year - my_df['yearbuilt'] )
    my_df['ordinal_date'    ] =  [x.toordinal() for x in my_df['transactiondate']]
    my_df['day_of_week'     ] = ( my_df['ordinal_date'] - my_df['ordinal_date'].min() + 2 ) % 7

    my_df['time_since_built'] = my_df['time_since_built'].fillna( my_df['time_since_built'].median() )

    my_df = my_df.drop( ['yearbuilt'], axis=1 )





    # Only flag which units have AC

    my_df['has_ac'] = 1

    my_df.ix[ my_df['airconditioningtypeid'].isnull(), ['has_ac'] ] = 0
    my_df.ix[ my_df['airconditioningtypeid'] == 5    , ['has_ac'] ] = 0

    my_df = my_df.drop( 'airconditioningtypeid', axis=1 )




    # Flag for has pool, has spa, size of pool

    my_df['has_spa'  ] = 0
    my_df['has_pool' ] = 0
    my_df['pool_sqft'] = 0

    my_df['has_spa'  ] = my_df['pooltypeid10'].fillna(0).astype(int)
    my_df['pool_sqft'] = my_df['poolsizesum' ].fillna(0)
    my_df['has_pool' ] =(my_df['poolsizesum' ] > 0).astype(int)

    my_df = my_df.drop( ['pooltypeid10','poolsizesum'], axis=1 )




    # Save the sqft in better variables

    my_df['unit_sqft'] = my_df['finishedsquarefeet12']
    my_df[ 'lot_sqft'] = my_df['lotsizesquarefeet'   ].fillna( my_df['lotsizesquarefeet'].median() )

    my_df = my_df.drop( ['finishedsquarefeet12', 'lotsizesquarefeet'], axis=1 )






    # n_units, if nan most likely a single unit. 1/nan, 2, 3, 4, multi

    my_df['unitcnt'].unique()

    my_df['unit_single'] = 0
    my_df['unit_double'] = 0
    my_df['unit_multi' ] = 0

    my_df.ix[ my_df['unitcnt'].isnull() , 'unit_single' ] = 1
    my_df.ix[ my_df['unitcnt'] == 1     , 'unit_single' ] = 1
    my_df.ix[ my_df['unitcnt'] == 2     , 'unit_double' ] = 1
    my_df.ix[ my_df['unitcnt'] >  2     , 'unit_multi'  ] = 1

    my_df = my_df.drop( 'unitcnt', axis=1 )





    # Flag for tax delinquency

    my_df['tax_delinquent'] = my_df['taxdelinquencyflag'].fillna(0).replace( {'Y': 1} )

    my_df = my_df.drop( 'taxdelinquencyflag', axis=1 )





    # Break building quality into categories

    my_df['building_quality_low' ] = 0
    my_df['building_quality_med' ] = 0
    my_df['building_quality_high'] = 0
    my_df['building_quality_unkn'] = 0

    my_df.ix[  my_df['buildingqualitytypeid'] < 4     , 'building_quality_high' ] = 1
    my_df.ix[  my_df['buildingqualitytypeid'] > 8     , 'building_quality_low'  ] = 1
    my_df.ix[ (my_df['buildingqualitytypeid'] > 3)    &
            (  my_df['buildingqualitytypeid'] < 9)    , 'building_quality_med'  ] = 1
    my_df.ix[  my_df['buildingqualitytypeid'].isnull(), 'building_quality_unkn' ] = 1

    my_df = my_df.drop( 'buildingqualitytypeid', axis=1 )





    # Have flag for garage, and sqft variable

    my_df['has_garage' ] = 0
    my_df['garage_sqft'] = my_df['garagetotalsqft'].fillna(0)

    my_df.ix[ my_df['garagetotalsqft']>1, 'has_garage' ] = 1

    my_df = my_df.drop( 'garagetotalsqft', axis=1 )






    # Break into single, multi-story flag

    my_df['story_single'] = 0
    my_df['story_multi' ] = 0
    my_df['story_unkn'  ] = 0

    my_df.ix[ my_df['numberofstories'] == 1    , 'story_single' ] = 1
    my_df.ix[ my_df['numberofstories'] >  1    , 'story_multi'  ] = 1
    my_df.ix[ my_df['numberofstories'].isnull(), 'story_unkn'   ] = 1


    my_df = my_df.drop( 'numberofstories', axis=1 )


    # 1 hot encode deck, var for number of fireplaces

    #my_df['n_fireplaces'] = my_df['fireplacecnt'].fillna(0)
    my_df['has_deck'    ] = my_df['decktypeid'  ].notnull().astype(int)

    my_df = my_df.drop( ['fireplacecnt','decktypeid'], axis=1 )



    # Drop any duplicate rows

    my_df = my_df.drop_duplicates( subset='parcelid', keep="last")


    # Do tax regression
    
    foo = regression_taxamount( my_df )

    my_df['log_tax_amount'] = foo['taxamount']
    my_df['log_tax_value' ] = foo['taxvaluedollarcnt']
    my_df['log_land_tax'  ] = foo['landtaxvaluedollarcnt']
    my_df['log_str_tax'   ] = foo['structuretaxvaluedollarcnt']

    del foo

    
    my_df['structuretaxvaluedollarcnt'] = 10**my_df['log_str_tax']
    my_df['n_bath'] = reg_num_bath( my_df )


    # Final steps

    my_df['n_rooms'] = my_df['roomcnt']

    
    my_df['unit_sqft'] = reg_unit_sqft( my_df )

    my_df['n_rooms'] = reg_rooms( my_df )


    return my_df.drop( ['calculatedbathnbr', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 
                     'landtaxvaluedollarcnt', 'taxamount', 'roomcnt', 'transactiondate', 'ordinal_date' ], axis=1 )


# Takes the normalization and PCA stuff from the scaling pickles and normalizes the df
def scale_data( inp_df ):

    scaled_df = inp_df.copy()
    scaled_df['age'] = scaled_df['time_since_built']
    
    scaled_df['day_of_week_norm'] = ( scaled_df['day_of_week'] - 3. ) / 3

    
    # Names of the dictionary
    dict_names = ['age','pool','unit','lot','garage']
    
    # Feature names
    feat_names = ['age','pool_sqft','unit_sqft','lot_sqft','garage_sqft']
    
    # List of the new features
    new_features = []
    
    # For each zscale feature:    
    for dic, feat in zip( dict_names, feat_names ):
        with open('data/pickle/'+dic+'_scale.pickle', 'rb') as handle:
            my_dict = pickle.load( handle )
            
            # If not age, need to log scale
            var_name = feat
            if ( feat != 'age' ):
                var_name = 'log_'+feat
                scaled_df[var_name] = np.log10( scaled_df[feat] + 1 )

            scaled_df[ var_name + '_scaled' ] = ( scaled_df[var_name] - my_dict['mean'] ) / my_dict['std']
            
            new_features.append( var_name + '_scaled' )

            
    # Min/max normalization, with some tweaking
    scaled_df['log_n_bath'] = np.log10( scaled_df['n_bath'] + 1 )

    with open('data/pickle/bath_scale.pickle', 'rb') as handle:
        my_dict = pickle.load( handle )
        scaled_df[ 'log_n_bath_scaled' ] = 2*( 2*( scaled_df['log_n_bath'] - my_dict['min'] ) / ( my_dict['max'] - my_dict['min'] ) - 1 )
        new_features.append( 'log_n_bath_norm' )

    with open('data/pickle/room_scale.pickle', 'rb') as handle:
        my_dict = pickle.load( handle )
        scaled_df[ 'n_rooms_scaled' ]    = 4*( 2*( scaled_df[   'n_rooms'] - my_dict['min']+1 ) / ( my_dict['max'] - my_dict['min'] ) - 1 )
        new_features.append( 'n_rooms_scaled' )
        
        
    tax_list = ['log_tax_amount', 'log_tax_value', 'log_land_tax', 'log_str_tax']
    tax_scal = ['log_tax_amount_scaled', 'log_tax_value_scaled', 'log_land_tax_scaled', 'log_str_tax_scaled']
    
    # Do tax PCA scaling, 4 features to 2
    with open('data/pickle/tax_pca.pickle','rb') as handle:
        tax_pca = pickle.load( handle )
        
        for item, sig in zip( tax_list, [3.0,3.0,2.0,2.5] ):
            scaled_df[item+'_scaled'] = rv.smart_scale( scaled_df, column=item, n_sigma=sig, show_final=False, curve_boost=4.5e4 )
        
        foo = tax_pca.transform( scaled_df[['log_tax_amount_scaled','log_tax_value_scaled','log_land_tax_scaled','log_str_tax_scaled']] )
        scaled_df['tax_pca_0'] = foo[:,0]
        scaled_df['tax_pca_1'] = foo[:,1]

        
    return scaled_df.drop( feat_names+tax_list+tax_scal+['time_since_built','day_of_week',
                            'log_pool_sqft','log_unit_sqft','log_lot_sqft','log_garage_sqft',
                            'n_bath','log_n_bath','n_rooms'], axis=1 )