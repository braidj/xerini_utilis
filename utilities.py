"""Common functions required across scripts"""
import ast
import csv
import os
import json
import codecs
import sys
import logging
import re
import pyperclip
import pandas as pd

class SingletonError(Exception):
    """To keep pylint happy"""

class MyLogger:
    
    """
    To use the logger, simply call MyLogger.get_logger() in any module where logging is needed.
    Only one instance of the logger will be created and shared between all modules.
    """
    __instance = None

    @staticmethod
    def getLogger():
        """Returns a single logger instance"""
        if MyLogger.__instance == None:
            MyLogger()
        return MyLogger.__instance.logger

    def __init__(self):
        if MyLogger.__instance != None:
            raise SingletonError("This class is a singleton!")
        else:
            MyLogger.__instance = self
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler('debug.log')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def reset_log(self):
        """Empty out the log file"""
        with open('debug.log', 'w',encoding='UTF-8'):
            pass

def check_csv_folder_exists(file_path,create,logger):
    """
    Checks if the file folder path exists and creates it if not
    Expects the full file path and the boolean create
    If True will simply create if required
    If False will exit with error if folder not present
    """
    logger.debug('check_csv_folder_exists received: %s',locals())

    directory_path = os.path.dirname(file_path)

    if not os.path.exists(directory_path):
        if create:
            os.makedirs(directory_path)
            logger.info('Had to created directory %s', directory_path)
        else:
            logger.error('Directory %s does not exist', directory_path)
            sys.exit(1)
    else:
        logger.debug('Directory %s exists', directory_path)


def csv_count_rows(folder_path,file_name,logger):
    """
    Counts the number of rows in a csv file
    """

    logger.debug('csv_count_rows received: %s',locals())
    source = os.path.join(folder_path,file_name +'.csv')

    with open(source, 'r') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)

    logger.debug('%s has %d rows.', source, row_count)

    return row_count


def csv_sort(folder_path,file_name,sort_by,logger):
    """
    Using Pandas sort the file by the columns supplied
    """

    logger.debug(f'csv_convert_dict_to_list received {locals()}')

    if type(sort_by) is not list:
        sort_by = sort_by.split(",")

    source_file = os.path.join(folder_path,file_name+ '.csv')

    data_frame = pd.read_csv(source_file)

    try:
        data_frame = data_frame.sort_values(by=sort_by).reset_index(drop=True)
        logger.debug("Sorted %s by %s", source_file, sort_by)
    except Exception as error:
        logger.error('An unexpected error occurred %s (%s)', file_name, str(error))

    data_frame.to_csv(source_file,encoding='utf-8', index=False)
    print(f"Sorted {source_file} has {len(data_frame)} rows")


def data_frame_sort_and_save(data_frame,destination_file,sort_by,logger):
    """
    Sorts the data_frame by the specified column(s)
    Changes the column order to match the sort_by list
    If columns are not all present then not sorted / ordered
    but is saved as a csv
    Param data types
    data_frame: pandas data frame
    destination_file: full file path string
    sort_by: list of column headers
    """
    
    logger.debug('csv_sort_and_save received: %s,%s',destination_file,sort_by)

    all_columns = data_frame.columns.tolist()

    if type(sort_by) is not list:
        sort_by = sort_by.split(",")

    try:
        sorted_df = data_frame.sort_values(by=sort_by).reset_index(drop=True)

        if all_columns == sort_by:
            sorted_df = sorted_df[sort_by] # re order columns

        sorted_df.to_csv(destination_file,encoding='utf-8', index=False)
        logger.debug("Sorted %s by %s", destination_file, sort_by)
    except KeyError:
        logger.debug("Unable to sort %s by %s", destination_file, sort_by)
        data_frame.to_csv(destination_file,encoding='utf-8', index=False)

def display_list_differences(list_a,list_b,label=""):
    """
    Show all differences between two lists
    Normally used to highlight where file headers differ
    """
    
    diff_a = list(set(list_a) - set(list_b))
    diff_b = list(set(list_b) - set(list_a))
    diff_all = list(set(list_a) ^ set(list_b))

    print(f"\nDifferences {label}")
    print("Elements in 'a' that are not in 'b':", diff_a)
    print("Elements in 'b' that are not in 'a':", diff_b)
    print("All unique elements that are different:", diff_all)

def csv_diff(details,logger):
    """
    Tool to compare 2 CSV files to highlight differences
    Used as part of testing to compare output of 
    Signify Utilities to the XEFR ETL layer
    Expects details dictionary:
        'file_a':'TSP UK Invoice Tracker',
        'file_a_path':TSP_DATA_DIR,
        'file_b':'TSP UK Invoice Tracker',
        'file_b_path':XEFR_DATA_DIR,
        'sort_by':'InvoiceDate,Placement,Candidate,CompanyName,Currency,LinePrice,ChargeCode,Country,Source',
        'reconciliation_folder': RECONCILIATION_DIR
    """

    test_pass = True
    skip_comparison = False
    fail_areas = []
    differences = []

    file_a = os.path.join(details['file_a_path'],details['file_a'] +'.csv')
    expected = os.path.join(details['reconciliation_folder'],details['file_a']+'_expected.csv')
    file_b = os.path.join(details['file_b_path'],details['file_b'] +'.csv')
    actual = os.path.join(details['reconciliation_folder'], details['file_b']+ '_actual.csv')

    sort_list = details['sort_by'].split(",")

    # Read files, save down to ensure are in same format, use
    # those copies as the sources for the comparison, store them down sorted as well
    # to aid visual comparisom

    data_frame_a = pd.read_csv(file_a)
    cols_file_a = data_frame_a.columns.tolist()
    data_frame_sort_and_save(data_frame_a,expected,sort_list,logger)

    data_frame_b = pd.read_csv(file_b)
    cols_file_b = data_frame_b.columns.tolist()
    data_frame_sort_and_save(data_frame_b,actual,sort_list,logger)
    
    data_frame_a = pd.read_csv(expected)
    data_frame_b = pd.read_csv(actual)

    # contain same columns, regardless of order
    have_same_columns = (sorted(cols_file_a) == sorted(cols_file_b))

    if have_same_columns == False: 
        test_pass = False
        skip_comparison= True
        fail_areas.append("Column headers")

    # Compare record count
    data_a_rows = len(data_frame_a)
    data_b_rows = len(data_frame_b)
    record_count_match = (data_a_rows == data_b_rows)

    if record_count_match == False: 
        test_pass = False
        fail_areas.append("Record count")

    record_count_diff = abs(data_a_rows - data_b_rows)

    # Sort data by supplied list of columns
    if have_same_columns:

        dfa_sorted = data_frame_a.sort_values(by=sort_list).reset_index(drop=True)
        dfb_sorted = data_frame_b.sort_values(by=sort_list).reset_index(drop=True)

        # Merge the DataFrames and highlight differences
        merged = dfa_sorted.merge(dfb_sorted, on=sort_list, how='outer', indicator=True)

        # Filter rows with differences
        differences = merged[merged['_merge'] != 'both']

        if len(differences) > 0:
            test_pass = False
            fail_areas.append("Row differences")

    col_order_same = (cols_file_a == cols_file_b)

    results = {
        'Test': details['file_a'],
        'Detail': f"Comparing {file_a} to {file_b}",
        'Test Passed': f"{test_pass} (failures = {fail_areas})",
        'Have same columns': have_same_columns,
        'Columns in same order': col_order_same,
        'Record Count Match': record_count_match,
        'Record Count Difference': f"{record_count_diff} ({details['file_a_path']} = {data_a_rows}, {details['file_b_path']} = {data_b_rows})",
        'Row Differences': len(differences)
    }

    print()
    for key in results:
        print(f"{key}: {results[key]}")

    if skip_comparison == False:
        if len(differences) > 0:
            print("\nFiltering to show only ten rows of differences")
            print(differences.head(5))
        else:
            print(f"\nWarning content could not be compared due to {fail_areas}")

    if have_same_columns == False:
        display_list_differences(cols_file_a,cols_file_b,"Column headers")

    if col_order_same == False:
        print("\nWarning: column order differs")
        print(f"file_a: {cols_file_a}")
        print(f"file_b: {cols_file_b}")

def csv_remove_duplicates(folder_path,file_name,logger):
    """
    Ensure a csv only contains unique rows
    """

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)
    original_count = len(data_frame)
    
    # Remove duplicates and keep only unique rows
    df_unique = data_frame.drop_duplicates()
    unique_count = len(df_unique)
    
    # Write the unique rows to a new CSV file
    df_unique.to_csv(original, index=False,encoding='utf-8')

    print(f"Removed {original_count - unique_count} rows, leaving {unique_count} unique rows")
    


def csv_convert_dict_to_list(folder_path,file_name,details,logger):
    """
    Extract out the elements of a dictionary and create a list of the values
    Details is a dictionary object e.g.
    {'column':'departments','list_key':'data','value_key':'name'}
    """

    logger.debug(f'csv_convert_dict_to_list received {locals()}')
    source = details['column']

    input_file = os.path.join(folder_path,file_name+ '.csv')
    output_file = os.path.join(folder_path,'temp_'+file_name+ '.csv')

    with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:

            data_dict =  ast.literal_eval(row[source])
            list_values = [x[details['value_key']] for x in data_dict[details['list_key']]]

            row[source] = list_values

            writer.writerow(row) # Write the modified row to the output file

    overwrite_file(output_file,input_file,logger,True)

def csv_extract_fullname_from_owners(folder_path,file_name,col_name,logger):
    """
    Extract full name from a Bullhorn owners data structure and return only the
    first item in the embedded list
    Need to pass in the column name to extract from
    """

    logger.debug('csv_extract_fullname_from_owners received: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    data_frame[col_name] = data_frame[col_name].apply(csv_extract_fullname_dict_list)
    data_frame.to_csv(original,encoding='utf-8', index=False)


def csv_extract_fullname_dict_list(value):
    """
    Extract full name from a data structure that is in this format
    "{'total': 1, 'data': [{'id': 93087, 'firstName': 'Michael', 'lastName': 'Hart'}]}"
    This is a special case for Bullhorn clients
    Selects only the first item in the embedded list
    This is called by csv_extract_fullname_from_owners()
    """

    data_dict = ast.literal_eval(value)

    if 'data' in data_dict and len(data_dict['data']) > 0:
        owner = data_dict['data'][0]
        first_name = owner.get('firstName')
        last_name = owner.get('lastName')

        return f"{first_name} {last_name}"
    else:
        return ""
        
def csv_remove_single_filter(folder_path,file_name,details,logger):
    """
    Removes all rows where the specified column has the specified value
    filter_col is the column to filter on
    filter_val is the value to filter on
    """

    logger.debug('csv_remove_single_filter received: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)
    original_count = len(data_frame)

    filtered_data_frame = data_frame[data_frame[details['filter_col']] != details['filter_val']]
    new_count = len(filtered_data_frame)

    filtered_data_frame.to_csv(original,encoding='utf-8', index=False)

    logger.info('Removed %s rows, leaving %s' % (original_count - new_count,new_count))

def csv_keep_single_filter(folder_path,file_name,details,logger):
    """
    Keeps all rows where the specified column has the specified value
    filter_col is the column to filter on
    filter_val is the value to filter on
    """

    logger.debug('csv_keep_single_filter received: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)
    original_count = len(data_frame)

    filtered_data_frame = data_frame[data_frame[details['filter_col']] == details['filter_val']]
    new_count = len(filtered_data_frame)

    filtered_data_frame.to_csv(original,encoding='utf-8', index=False)

    print(f"Removed {original_count - new_count} rows, leaving {new_count} where {details['filter_col']} == {details['filter_val']}")
    logger.info('Removed %s rows, leaving %s' % (original_count - new_count,new_count))


def return_keys(value,keys):
    """
    Used by csv_extract_keys to extract the required keys
    """

    try:
        dictionary_data = ast.literal_eval(value)
        result = [dictionary_data[x] for x in dictionary_data if x in keys]
    except ValueError:
        result = ['','']

    return result

def csv_extract_keys(folder_path,file_name,details,logger):
    """
    For embedded json columns, extract the keys and create a new column for each
    """
    
    logger.debug('csv_extract_keys received: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')

    data_frame = pd.read_csv(original, keep_default_na=False, na_values={details['source']: ''})

    new_col_vals = data_frame[details['source']].apply(lambda x: return_keys(x,details['required_keys']))

    data_frame[details['new_cols']] = pd.DataFrame(new_col_vals.tolist(), index= data_frame.index)
    data_frame.drop(details['source'], axis=1, inplace=True)
    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_fill_in_nulls(folder_path,file_name,details,logger):
    """Add a default value for the specified columns when null"""

    logger.debug('csv_fill_in_nulls received: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    for item in details:
        col = item
        value = details[item]
        data_frame.loc[data_frame[col].isnull(), col] = value

    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_remove_dec_places(folder_path,file_name,dec_cols,logger):
    """
    Removes the decimal places from the specified columns
    dec_cols is a comma separated list of columns
    """

    logger.debug('csv_remove_dec_places received: %s',locals())
    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    for col in dec_cols.split(","):
        data_frame[col] = data_frame[col].astype(int)

    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_remove_null_cols(folder_path,file_name,details,logger):
    """
    Reduce the file down, removing any specified columns that are empty
    Additionally remove any duplicate rows if passed
    Details = {empty_cols: [List of cols], drop_duplicates: True/False}
    """
    original = os.path.join(folder_path,f'{file_name}.csv')
    logger.debug('csv_remove_null_cols received: %s',locals())

    dataframe = pd.read_csv(original)
    original_count = len(dataframe)
    logger.info('%s has %s rows', file_name, original_count)

    dataframe = dataframe.dropna(subset=details['empty_cols'])

    dataframe = dataframe.reset_index(drop=True) # Reset the index after dropping rows
    filtered_count = len(dataframe)
    logger.info('Removing Nulls on column %s: %s now has %s rows, a difference of %s',details['empty_cols'], file_name, filtered_count, original_count - filtered_count)

    if details['drop_duplicates']:
        dataframe = dataframe.drop_duplicates()
        unique_count = len(dataframe)
        logger.info('Dropping anys duplicates: %s now has %s rows, a difference of %s', file_name, unique_count, filtered_count - unique_count)

    logger.info('DEBUG: %s', len(dataframe))

    dataframe.to_csv(original,encoding='utf-8', index=False)

def xefr_data_api(api_token,schema_id,desc,end_point='http://localhost:7080/api/data/format',format='csv'):
    """
    Returns the curl command to return data
    from the XEFR data API in the required format
    N.B. it does run it, just returns the command
    """

    if format not in ['csv','json']:
        print(f"Format {format} not supported")
        sys.exit(1)

    curl_command = f'curl -H "X-API-KEY: {api_token}" {end_point}/{format}/{schema_id}'
    print(f"{desc}: {curl_command}")
    save_to_clipboard(curl_command)
 
def compare_tsp_bullhorn_users(folder_path_1,file_one,folder_path_2,file_two,logger):

    """Compare two csv files and return the differences"""

    logger.debug('csv_compare_files received: %s',locals())
    source_1 = os.path.join(folder_path_1,file_one +'.csv')
    source_2 = os.path.join(folder_path_2,file_two +'.csv')
    data_frame_1 = pd.read_csv(source_1)
    data_frame_2 = pd.read_csv(source_2)

    data_frame_2['id'] = data_frame_2['id'].astype(str)
    merged_df = pd.merge(data_frame_1, data_frame_2, left_on=['userCode'], right_on=['id'])

    result_df = merged_df[['userCode','fullName_x', 'fullName_y']]

    result_df.columns = ['userCode','TSP_fullName', 'Bullhorn_fullName']
    result_df['difference'] = 'No'
    result_df.loc[result_df['TSP_fullName'] != result_df['Bullhorn_fullName'], 'difference'] = 'Yes'
    result_file = os.path.join(folder_path_1,'tsp_bullhorn_user_compare.csv')
    result_df.to_csv(result_file, encoding='utf-8',index=False )

def csv_add_default_column(folder_path,file_name, details,logger):
    """
    Adds a default columns and value to the csv file
    """

    logger.debug('csv_add_default_column: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')

    data_frame = pd.read_csv(original)

    for key,value in details.items():
        data_frame[key] = value

    try:
        headers = data_frame.columns.tolist()
        data_frame.to_csv(original, encoding='utf-8',index=False)
        logger.debug("Added column(s) %s to %s", details.keys(), headers)

    except PermissionError as error:
        logger.error('An unexpected error occurred %s (%s)', file_name, str(error))
        sys.exit(1)

    

def csv_distinct_values(folder_path,file_name, col_name,logger):
    """
    Returns a sorted list to terminal of the distinct values for the column
    """

    logger.debug('csv_distinct_values: %s',locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original,encoding='UTF-8')
    distinct_values = data_frame[col_name].unique()

    # # Read the CSV file with the appropriate encoding and error handling
    # data_frame = pd.read_csv(file_path, encoding='unicode_escape', error_bad_lines=False)
    # # Get the distinct values from the currency column
    # distinct_values = data_frame['currency_column'].unique()

    print(distinct_values)
    return distinct_values

def csv_split(folder_path,file_name, chunk_size,logger):
    """Splits a csv file into chunks of the designated size"""

    logger.debug('csv_split received: %s',locals())
    original = os.path.join(folder_path,file_name +'.csv')

    with open(original, 'r',encoding='UTF-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Assuming the first row is the header
        chunk_number = 1
        row_count = 0
        current_chunk = []

        for row in reader:
            current_chunk.append(row)
            row_count += 1

            if row_count == chunk_size:
                output_file = os.path.join(folder_path,f"{file_name}_{chunk_number}.csv")
                write_chunk_to_csv(header, current_chunk, output_file)
                chunk_number += 1
                current_chunk = []
                row_count = 0

            # Write the remaining rows as the last chunk (if any)
            if row_count > 0:
                output_file = os.path.join(folder_path,f"{file_name}_{chunk_number}.csv")
                write_chunk_to_csv(header, current_chunk, output_file)

def write_chunk_to_csv(header, chunk, output_file):
    """allows a file to be split up"""

    with open(output_file, 'w',encoding='UTF-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(chunk)

def csv_pad_data(folder_path,file_name,attributes,logger):
    """"
    Add blank rows of data so that entire length of time series is accessible
    This is done before the data is loaded into PowerBI
    Must use an actual record to pad data out though
    """
    logger.debug('csv_pad_data received: %s', locals())

    month_nos = ['01','02','03','04','05','06','07','08','09','10','11','12']
    month_start_series = [f"{attributes['year']}-{x}-01" for x in month_nos]

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    search_col = attributes['search_for']['col_name']
    search_val = attributes['search_for']['col_value']

    if '!' in search_val:
        val = search_val.replace('!',"")
        mask = data_frame[search_col] == val
    else:
        mask = data_frame[search_col] != search_val

    key_col = attributes['key_column']
    first_matching_row = data_frame.loc[mask].iloc[0]
    key_col_value = first_matching_row[key_col]

    new_row_template = attributes['mandatory_columns']
    new_row_template[key_col] = key_col_value

    for item in month_start_series:

        row = new_row_template
        row[attributes['date_col']] = item # extend the data to append
        new_row = pd.DataFrame(row, index=[0])
        data_frame = pd.concat([data_frame, new_row], ignore_index=True)

    data_frame.to_csv(original,encoding='utf-8', index=False)

def cleanse_string_to_decimal(value):
    """
    Converts string with currency codes in them to plain decimals
    """

    value = value.replace(',', '')

    numeric_string = re.sub(r'[^0-9.-]', '', value)
    decimal_value = float(numeric_string)

    return decimal_value

def csv_change_column_order(folder_path,file_name,col_order,logger):
    """
    Changes the order of the columns in the file
    """

    logger.debug('csv_change_column_order received: %s', locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    desired_order = col_order.split(",")

    data_frame = data_frame[desired_order]

    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_remove_currency_symbol(folder_path,file_name,columns,logger):
    """
    Processes the columns to extract out the currency symbol
    this is an in place replacement.
    """
        
    logger.debug('csv_remove_currency_symbol received: %s', locals())

    targets = columns.split(",")    
    
    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    for col in targets:
        data_frame[col] = data_frame[col].apply(extract_numbers_from_string)
    
    data_frame.to_csv(original,encoding='utf-8', index=False)


def extract_currency_text(currency_string):
    """
    Return the currency symbol portion of a currency amount
    Typically used in a pandas apply function
    """
    if type(currency_string) == float:
        return ""
    
    match = re.search(r'\d',currency_string)

    if match:
        number_start = match.start()
        return currency_string[0:number_start]
    else:
        return(currency_string)
    
def extract_numbers_from_string(currency_string):
    """
    Returns just the numbers of a currency string
    Expect format to be currency symbol followed by numbers
    """
    if type(currency_string) == float:
        return ""

    match = re.search(r'\d',currency_string)
    if match:
        number_start = match.start()
        return str(currency_string[number_start:])

    else:
        return(currency_string)

def csv_add_currency_symbol(folder_path,file_name,details,logger):
    """
    Processes the columns to extract out the currency symbol
    and create a new column with it.
    """
    logger.debug('csv_add_currency_symbol received: %s', locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    for source, new_col in details.items():

        data_frame[new_col]=data_frame[source].apply(extract_currency_text)
    
    data_frame.to_csv(original,encoding='utf-8', index=False)


def csv_convert_to_decimal(folder_path,file_name,columns,logger):
    """
    Convert specified columns into decimals
    """
    logger.debug('csv_convert_to_decimal: %s', locals())

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    for col in columns.split(","):

        data_frame[col] = data_frame[col].apply(cleanse_string_to_decimal)

    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_concatenate_columns(folder_path,file_name,actions,logger):
    """
    Create new column based on two existing columns with a delimiter
    """
    logger.debug('csv_concatenate_columns: %s', locals())
    original = os.path.join(folder_path,file_name +'.csv')

    data_frame = pd.read_csv(original)

    data_frame[actions['new_col']] = (
        data_frame[actions['col_list'][0]]
        + actions['delimiter']
        + data_frame[actions['col_list'][1]]
        )
    
    if actions['remove_cols']:
        data_frame = data_frame.drop(actions['col_list'], axis=1)

    data_frame.to_csv(original,encoding='utf-8', index=False)

def csv_extract_columns(folder_path,file_name,columns_to_keep,new_names,logger):
    """
    Reduce the file down to the selected columns
    Rename them
    """

    logger.debug('csv_extract_colmns received: %s', locals())

    if len(columns_to_keep.split(",")) != len(new_names.split(",")):
        print("Number of columns to keep and new names must match")
        logger.error('Number of columns to keep and new names must match')
        sys.exit(1)

    original = os.path.join(folder_path,file_name +'.csv')
    data_frame = pd.read_csv(original)

    columns_to_keep = columns_to_keep.split(",")
    df_filtered = data_frame[columns_to_keep]

    if new_names != "no":
        new_col_names = new_names.split(",")
        df_filtered.columns = new_col_names

    df_filtered.to_csv(original,encoding='utf-8',index=False)

def start_end_months(year,logger):
    """"
    Returns a list of of start and end dates for each month
    Useful when querying by a month
    """

    logger.debug('start_end_months recevied: %s', locals())

    time_suffix = 'T00:00:00Z'

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_nos = ['01','02','03','04','05','06','07','08','09','10','11','12']
    nos_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    results = []

    for i in month_labels:

        idx = month_labels.index(i)

        results.append(
            {
                    'month_label' : i,
                    'month_nos': month_nos[idx],
                    'startDate':f'{year}-{month_nos[idx]}-01{time_suffix}',
                    'endDate':f'{year}-{month_nos[idx]}-{nos_days[idx]}{time_suffix}'
                    }
        )

    return results

def csv_list_of_lists(folder_path,data,file_name,logger):
    """"
    Save a list of lists as a csv file
    data:       list of lists,expect first row to be the headers
                TSP Reports are in this format
    file_name:  file / report name, without .csv
    logger:     generic logger
    
    """
    ignore_keys = ['data',]
    params_recevied = [f'{k}: {v}' for k,v in locals().items() if k not in ignore_keys]
    logger.debug("csv_list_of_lists received: %s",params_recevied)

    source = os.path.join(folder_path,file_name+'.csv')

    try:
        with open(source, 'w', encoding='UTF-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
            print(f"Saved {file_name}.csv to {folder_path} ({len(data)} rows)")
            return True

    except PermissionError as error:
        logger.error('No access to file %s (%s)', source, str(error))

    except FileNotFoundError as error:
        logger.error('File not found %s (%s)', source, str(error))

    except Exception as error:
        logger.error('An unexpected error occurred %s (%s)', source, str(error))

    print(f"Unable to save {file_name}.csv to {folder_path}, check the debug.log file for details")
    return False


def csv_convert_dates(folder_path,csv_name,date_columns,input_format, output_format,logger):
    """Converts all the dates in a csv file to the specified format"""
    logger.debug(f'csv_convert_dates recevied: {locals()}')
    #'%Y-%m-%d'

    original = os.path.join(folder_path,csv_name+'.csv')
    data_frame = pd.read_csv(original)

    for col in date_columns:
        data_frame[col] = pd.to_datetime(data_frame[col],format = input_format)
        data_frame[col] = data_frame[col].dt.strftime(output_format)

    data_frame.to_csv(original,encoding='utf-8', index=False)

def check_dictionary_values(dictionary, lst, exception_item):
    """Returns true if any values of a dictionary exist in the column header list"""

    return any(value in lst for value in dictionary.values() if value != exception_item)

def csv_add_datetime_cols(folder_path,csv_name,actions,logger,unit_value='ms'):
    """
    Adds formatted date column and or time based on a ts column
    Can specifiy whether to remove the ts column
    Also Checks if new columns have already been added
    """

    logger.debug(f'csv_add_datetime_columns received: {locals()}')
    original = os.path.join(folder_path,csv_name+'.csv')

    data_frame = pd.read_csv(original)
    headers_list = data_frame.columns

    if check_dictionary_values(actions,headers_list,actions['ts_column']):
        logger.info(f'Will not add new date, time columns for {csv_name} as they already exist')
    else:

        #For unit value ms for unix time, ns for UTC
        ts_column = actions['ts_column']
        data_frame[ts_column] = pd.to_datetime(data_frame[ts_column], unit=unit_value)

        if 'date_name' in actions:
            col = actions['date_name']
            data_frame[col] = pd.to_datetime(data_frame[ts_column],format = actions['input_format'])

            if actions['output_format'] == '%q':
                data_frame[col] = data_frame[col].dt.quarter
            else:
                data_frame[col] = data_frame[col].dt.strftime(actions['output_format'])

        if 'time_name' in actions:
            data_frame[actions['time_name']] = data_frame[ts_column].dt.strftime('%H:%M:%S')

        if actions['remove_ts_column']:
            data_frame = data_frame.drop(ts_column, axis=1)

        data_frame.to_csv(original,encoding='utf-8', index=False)

def change_csv_column_order(folder_path,csv_name,column_order,logger):
    """Have a wild stab in the dark as to what this does"""

    logger.debug(f'change_csv_column_order recevied: {locals()}')
    original = os.path.join(folder_path,csv_name)

    data_frame = pd.read_csv(original)
    data_frame = data_frame[column_order]
    data_frame.to_csv(original,encoding='utf-8', index=False)

def overwrite_file( source, target,logger,remove_source=False,):
    """Overwrite the target file with the cotents of the source file"""

    logger.debug(f'overwrite_file recevied: {locals()}')

    with open(target, 'w',encoding='UTF-8') as out_file,\
        open(source, 'r',encoding='UTF-8') as in_file:
        out_file.write(in_file.read())

    if remove_source:
        os.remove(source)


def json_join_attributes(folder_path, file_name, details, logger):
    """
    Join required 2 attributes from a document, and store as a new field
    if there are more than 1 set, use the first and warn
    """

    logger.debug(f'json_join_attributes received: {locals()}')

    original = os.path.join(folder_path, file_name + '.json')
    updated = os.path.join(folder_path,'temp_' + file_name + '.json')
    proceed = True

    with open(original,'r',encoding='UTF-8') as json_original:

        data_rows= json.load(json_original)
        filtered_rows = []

        for row in data_rows:

            payload = row[details['document']]

            if payload['total'] != 1:
                logger.warning(f"More than 1 {details['new_column']} for {details['document']}: {payload['data']}")
            elif payload['total'] == 0:
                value = ""
            else:
                first, secound = details['attributes'].split(",")

                source = payload['data'][0]
                value = source[first] + details['delimiter'] + source[secound]

            row.pop(details['document'])
            row[details['new_column']] = value
            filtered_rows.append(row)


        with open(updated, "w",encoding='UTF-8') as result:
            json.dump(filtered_rows, result)

        overwrite_file(updated,original,logger,True)

def extract_key_json_list(folder_path, json_name,extract_from,required_key,logger):
    """Pull out a key you need from a list of json dictionaries"""
    logger.debug(f'extract_keys_json_dict received {locals()}')

    original = os.path.join(folder_path,json_name+ '.json')
    filtered = os.path.join(folder_path,'temp_'+json_name+ '.json')
    proceed = True

    with open(original,'r',encoding='UTF-8') as json_original:

        data_rows= json.load(json_original)
        filtered_rows = []

        for row in data_rows:
            try:
                dictionary_list = row[extract_from]['data']
                result = [i[required_key] for i in dictionary_list]
                row.pop(extract_from)
                row[extract_from] = result
                filtered_rows.append(row)
            except TypeError as error:
                logger.warning(str(error))
                logger.info('Looks like this data has already been refined')
                proceed = False
                break

    if proceed:
        with open(filtered, "w",encoding='UTF-8') as result:
            json.dump(filtered_rows, result)#indent=4

        overwrite_file(filtered,original,logger,True)

def extract_keys_json_dict(folder_path, json_name,extract_from,required_keys,logger):
    """
    Extract from a json document dictionary the required keys, remove the dictionary,
    add the required keys as new fields.
    Its a way of unflattening a json dict, but just for the keys you need
    """

    logger.debug(f'extract_keys_json_dict received {locals()}')

    original = os.path.join(folder_path,json_name+ '.json')
    filtered = os.path.join(folder_path,'temp_'+json_name+ '.json')
    proceed = True

    with open(original,'r',encoding='UTF-8') as json_original:
        data_rows= json.load(json_original)
        filtered_rows = []

        for row in data_rows:
            try:
                record = row[extract_from]
                for i in required_keys:
                    row[f'{extract_from}.{i}'] = record.get(i, '') # add the new values


                row.pop(extract_from)
                filtered_rows.append(row)

            except(KeyError) as error:

                logger.warning(f'Key {extract_from} does not exist {str(error)}')
                # proceed = False
                # break
            except AttributeError as error:
                logger.warning(f'document {extract_from} does not exist {str(error)}')
                # proceed = False
                # break

    if proceed:
        with open(filtered, "w",encoding='UTF-8') as result:
            json.dump(filtered_rows, result)#indent=4

        overwrite_file(filtered,original,logger,True)


def unique_values_csv(folder_path,csv_name,col_to_check,logger):
    """
    Returns to Terminal a unique list of values for the specified column in the csv file
    Just pass the file name not the extension
    """

    logger.debug(f'unique_values_csv recevied: {locals()}')
    csv_target = os.path.join(folder_path,csv_name+'.csv')

    with open(csv_target,'r', newline='',encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)

        if col_to_check not in reader.fieldnames:
            print(f"Error: Column '{col_to_check}' does not exist in the CSV file.")
        else:

            unique_values = set()
            for row in reader:
                unique_values.add(row[col_to_check])
            sorted_values = sorted(unique_values)

            for value in sorted_values:
                print(value)

def filter_json_list(data_folder, json_data, report,required_fields,logger):
    """
    Extract required fields out of a list of json records and
    persist to a file.
    
    Parameters:
    data_folder (str): data folder
    json_data (List[Dict[str, str]]): the full json file to be filtered
    report (str): base name of the file to be created (no ext)
    required_fields (str): comma separated string of fields to keep, optional
    logger (logging.Logger): a logger object to log debug and info messages

    Returns:
    str: Name of the file created
    """

    ignore_keys = ['logger', 'json_data']
    params_recevied = [f'{k}: {v}' for k,v in locals().items() if k not in ignore_keys]
    logger.debug("filter_json_list received: %s",params_recevied)

    if required_fields == '*':
        required_fields = None

    if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
        raise TypeError("json_data must be a list of dictionaries")

    new_file = os.path.join(data_folder,report + '.json')

    if required_fields is not None:
        if not isinstance(required_fields, str):
            raise TypeError("required_fields must be a string")
        keys_to_keep = required_fields.split(',')
        logger.debug("Filtering json_data to include fields: %s", keys_to_keep)
        data = [{k:v for k,v in record.items() if k in keys_to_keep} for record in json_data]
    else:
        logger.info("No filtering json_data to include fields")
        data = json_data


    with open(new_file ,'w',encoding='UTF-8') as json_file:
        json.dump(data,json_file)

    num_records = len(data)
    logger.info("%s created with %d records", new_file, num_records)

    return new_file

def filter_json(data_folder, original_json, filtered_name, required_fields,logger):
    """Create a new json file only containing the required fields
    required_fields is an array of strings"""
    logger.debug('filter_jason: recevived (excluding orignal_json) %s %s %s',
                  data_folder,filtered_name,required_fields)

    original_file = os.path.join(data_folder,original_json)
    new_file = os.path.join(data_folder,filtered_name)

    filtered_data=[]
    try:
        with open(original_file,'r',encoding='UTF-8') as original, \
                open(new_file,'w',encoding='UTF-8') as new:
            data = json.load(original)
            filtered_data = [{field: item[field] for field in required_fields} for item in data]
            json.dump(filtered_data,new)
    except (FileNotFoundError,json.JSONDecodeError) as error:
        print(f"An error occurred while filtering the JSON file: {error}")

    return filtered_data

def save_list_to_json(data_list,file_path,file_name,confirmation,logger):
    """Saves a list of json records to a file"""

    logger.debug('save_list_to_json recevied: (excl data_list) %s %s %s',
                 file_path,file_name,confirmation)

    json_input = os.path.join(file_path,f"{file_name}.json")

    with open(json_input,'w',encoding='UTF-8') as json_file:
        json.dump(data_list,json_file)

    if confirmation:
        print(f"{len(data_list)} rows in {json_input}")

    return json_file.name

def save_to_clipboard(text):
    """Saves the text to clipboard for manual pasting"""
    pyperclip.copy(text)

def json_to_csv(folder_path, file_name,logger):
    """Convert json file to csv of same name"""

    json_input= os.path.join(folder_path,file_name + '.json')
    csv_output = json_input.replace(".json",".csv")

    logger.debug('json_to_csv received: %s', locals())

    with open(json_input, 'r', encoding='UTF-8') as json_file, \
        open(csv_output, 'w', newline='', encoding='UTF-8') as csv_file:

        try:
            data = json.load(json_file)
            writer = csv.writer(csv_file)
            writer.writerow(data[0].keys())

            # Write each data row
            for row in data:
                writer.writerow(row.values())

        except IndexError:
            logger.info('Looks like the file has no data')
        except Exception as error:
            logger.error('An unexpected error occurred %s (%s)', file_name, str(error))
            sys.exit(1)

        logger.info(f"{json_input} converted to {csv_output}")

def json_validator(full_file_path,logger):
    """utility to confirm if a json file has been formatted properly"""

    logger.debug('json_validator recevied: %s', full_file_path)

    try:
        with open(full_file_path, 'r',encoding='UTF-8') as file:
            json_data = json.load(file)
            if isinstance(json_data, list):
                print(full_file_path, f"has {len(json_data)} records")
    except json.JSONDecodeError as error:
        print(full_file_path, "is NOT valid")
        print(str(error))
    except FileNotFoundError:
        print(full_file_path, "does not exist")

def csv_to_json(folder_path,csv_file,logger):
    """
    Utility to convert csv data into json, helps with mocking up TSP data
    Pass the file name WITHOUT the extension
    """

    logger.debug('csv_to_json recevieved: %s', locals())

    csv_data = os.path.join(folder_path,csv_file + '.csv')
    json_data = csv_data.replace('.csv','.json')

    try:
        with codecs.open(csv_data, 'r',encoding='utf-8-sig') as csvfile:

            # Parse the CSV data using the csv library
            csvreader = csv.DictReader(csvfile)

            # Convert each row in the CSV to a dictionary
            rows = []
            for row in csvreader:
                rows.append(row)

    except Exception as error:

        if "No such file or directory" in str(error):
            print(f"ERROR: {csv_data} does not exist.")
            return 1
        else:
            raise error

    # Write the resulting data as JSON to an output file
    with open(json_data, 'w',encoding='UTF-8') as jsonfile:
        json.dump(rows, jsonfile)
