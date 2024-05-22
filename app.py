#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:





# In[4]:


import openai
import psycopg2
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, render_template_string, jsonify


# In[5]:






def generate_sql_query(schema, user_query):
    prompt = f"""
    Given the following PostgreSQL table schema:
    {schema}
    
    And the user query: "{user_query}"

    Write the corresponding SQL query.also just give me only query to execute on postgres with out any description
    """
    
    #print(prompt)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates natural language queries into SQL queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"

    
    
def generate_sql_response(columns,rows):
    prompt = f"""
    I have below result by running a query on table which has some columns and data
    {columns,rows}
    
    Display the results in the natural language for the user to understand easily
    
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates natural language queries into SQL queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"    
    
    
    
def suggest_chart_type(columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    data_summary = df.describe(include='all').to_dict()

    prompt = f"""
    Given the following data summary:
    {data_summary}

    Suggest the most suitable type of chart for visualizing this data. Possible chart types include bar chart, line chart, scatter plot, pie chart, histogram, and box plot. Just provide the name of the chart without any explanation.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests the most suitable chart types for visualizing data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip().lower().replace(" chart", "")
    except Exception as e:
        return f"An error occurred: {e}"

    


# In[8]:


app = Flask(__name__)

def get_table_schema(conn, table_name):
    query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}';
    """
    with conn.cursor() as cur:
        cur.execute(query)
        schema = cur.fetchall()
    return {table_name: schema}

def connect_to_postgres():
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='vpatil01',
        host='localhost',
        port='5432'
    )
    return conn

def execute_sql_query(conn, sql_query):
    with conn.cursor() as cur:
        try:
            cur.execute(sql_query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return columns,rows
        except psycopg2.Error as e:
            return None, f"SQL execution error: {e}"


def create_and_display_suggested_chart(chart_type, df, image_name):
    plt.figure(figsize=(20, 10))

    if chart_type == "bar":
        categorical_col = df.select_dtypes(include=['object']).columns[0]
        numerical_col = df.select_dtypes(include=['number']).columns[0]
        df.groupby(categorical_col)[numerical_col].sum().plot(kind='bar', color=['blue', 'orange'])
        plt.title(f'Total {numerical_col} by {categorical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'Total {numerical_col}')
        plt.xticks(rotation=0)
    elif chart_type == "line":
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) >= 2:
            df.plot(kind='line', x=numerical_cols[0], y=numerical_cols[1])
            plt.title(f'Line plot of {numerical_cols[1]} over {numerical_cols[0]}')
            plt.xlabel(numerical_cols[0])
            plt.ylabel(numerical_cols[1])
    elif chart_type == "scatter":
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) >= 2:
            df.plot(kind='scatter', x=numerical_cols[0], y=numerical_cols[1])
            plt.title(f'Scatter plot of {numerical_cols[1]} vs {numerical_cols[0]}')
            plt.xlabel(numerical_cols[0])
            plt.ylabel(numerical_cols[1])
    elif chart_type == "pie":
        categorical_col = df.select_dtypes(include=['object']).columns[0]
        df[categorical_col].value_counts().plot(kind='pie')
        plt.title(f'Pie chart of {categorical_col}')
    elif chart_type == "histogram":
        numerical_col = df.select_dtypes(include=['number']).columns[0]
        df[numerical_col].plot(kind='hist', bins=20)
        plt.title(f'Histogram of {numerical_col}')
        plt.xlabel(numerical_col)
        plt.ylabel('Frequency')
    elif chart_type == "box plot":
        numerical_col = df.select_dtypes(include=['number']).columns[0]
        df[numerical_col].plot(kind='box')
        plt.title(f'Box plot of {numerical_col}')
    else:
        print(f"The suggested chart type '{chart_type}' is not supported.")

    plt.savefig('static/' + image_name)

    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    #buf.close()
    
    #return img_base64

 


# In[6]:


from flask import Flask, request, render_template_string, render_template, url_for
import logging
import os
from datetime import datetime
import traceback

app = Flask(__name__)
logging.basicConfig(filename='app.log', level=logging.INFO)

# Define your functions here (connect_to_postgres, generate_sql_query, etc.)

@app.route('/')
def home():
    return render_template('index.html')

#delete old images - not working though
def delete_images_in_static_folder():
    static_folder = 'static'
    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.png'):
            os.remove(file_path)

@app.route('/query', methods=['POST'])
def query():
    form_data = request.json
    user_query = ' '.join(word.capitalize() for word in form_data['q'].split());
    
    selected_dataset = form_data['dataset'];
    table_name = 'customer_properties'
    if selected_dataset and selected_dataset.strip():
        table_name = selected_dataset;

        
    conn = connect_to_postgres()    

    original_image_name = 'chart.png'

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Append timestamp to image name
    new_image_name = f'{os.path.splitext(original_image_name)[0]}_{timestamp}{os.path.splitext(original_image_name)[1]}'
    
    try:
        schema = get_table_schema(conn, table_name)
        sql_query = generate_sql_query(schema, user_query)
        logging.info(sql_query)
        columns, rows = execute_sql_query(conn, sql_query)
        response = generate_sql_response(columns, rows)

        chart_type = suggest_chart_type(columns, rows)
        df = pd.DataFrame(rows, columns=columns)
        create_and_display_suggested_chart(chart_type, df, new_image_name)

        image_url = url_for('static', filename=new_image_name)
        logging.info(chart_type)
        result = {
            "query": sql_query,
            "response": response,
            "image_url": image_url,
            "chart_type": chart_type
        }
    except Exception as e:
        # Log the full traceback to the console
        print(traceback.format_exc())
        result = {
            "query": "An error occurred while processing your query.",
            "response": str(e),
            "chart_type": None
        }
    finally:
        conn.close()

    return jsonify(result)

@app.route('/charts', methods=['GET'])
def charts():
    user_query = request.args.get('q')
    table_name = 'customer_properties'
    conn = connect_to_postgres()

    try:
        schema = get_table_schema(conn, table_name)
        sql_query = generate_sql_query(schema, user_query)
        columns, rows = execute_sql_query(conn, sql_query)
        response = generate_sql_response(columns, rows)

        chart_type = suggest_chart_type(columns, rows)
        df = pd.DataFrame(rows, columns=columns)
        create_and_display_suggested_chart(chart_type, df)

    except Exception as e:
        # Log the full traceback to the console
        print(traceback.format_exc())
        result = {
            "query": "An error occurred while processing your query.",
            "response": str(e),
            "chart_type": None
        }
    finally:
        conn.close()

    return render_template('charts.html')

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


def main():
    table_name = 'customer_properties'
    user_query = "show me the count of Males and Females in the gender"
    conn = connect_to_postgres()

    try:
        schema = get_table_schema(conn, table_name)
        sql_query = generate_sql_query(schema, user_query)
        #print("Generated SQL Query:")
        print(sql_query)
        
        columns,rows = execute_sql_query(conn, sql_query)
        #description = describe_results(columns, rows)
        
        #print("Query Results:")
        print(columns,rows)
        print(generate_sql_response(columns,rows))
        
        
        chart_type = suggest_chart_type(columns, rows)
        df = pd.DataFrame(rows, columns=columns)
        create_and_display_suggested_chart(chart_type, df)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()


# In[ ]:




