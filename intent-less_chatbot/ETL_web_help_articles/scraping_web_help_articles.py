"""
This script was used to scrape the web help articles and store them as .txt files.
The web help articles were subsequently saved in a Chroma vector database
(located at '\intent-less_chatbot\webhelp_and_websupport_vector_db') together with the web support Q&As from the
ticketing dataset
"""


# 1. Set up--------------------------------------------------------------------------------------------------------------
# Set path to root directory and intent-less chatbot folder
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis"
intent_less_path = path + r'\intent-less_chatbot'

# Import packages
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import os


# 2. Create a list of pages to scrape and store them in keyword-link-pairs.txt------------------------------------------
# Specify the directory where the ChromeDriver executable is located
chromedriver_path = r'C:\Program Files (x86)\chromedriver.exe'

# Initialize the ChromeDriver service
chrome_service = Service(executable_path=chromedriver_path)

# Initialize the WebDriver with headless option
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=chrome_service, options=options)

# Load the webpage containing HTML content
driver.get('https://applsupport.hslu.ch/webhelp/hmkwindex.htm')

# Getting the dynamic keywords and corresponding link
# Find the keyword element and click on it to load the content
keyword_element = driver.find_element(By.XPATH, "//a[contains(@onclick, \"return hmshowLinks('k1')\")]")
ActionChains(driver).click(keyword_element).perform()

# Wait for the content element to be visible
wait = WebDriverWait(driver, 10)
k_value = keyword_element.get_attribute('onclick').split("'")[1]
wait.until(EC.visibility_of_element_located((By.ID, k_value)))

# Find the content table with class 'idxtable'
content_table = driver.find_element(By.XPATH, "//table[@class='idxtable']")

# Find all <a> elements with class 'idxlink' within the content table
target_a_tags = content_table.find_elements(By.XPATH, "//table[@style='border:none;padding:0']//a")

dynamic_dict = {}
# Extract and store the text and href attributes from the <a> elements with class 'idxlink'
for idxlink_element in target_a_tags:
    # Use JavaScript to get the innerText of the <a> element
    text = driver.execute_script('return arguments[0].innerText;', idxlink_element)
    href = idxlink_element.get_attribute('href')
    dynamic_dict[text] = href

# Print the dictionary
print(dynamic_dict)


# Get all keywords and links from static content
# Find all <a> elements within <p> tags with the specified class
a_elements = driver.find_elements(By.XPATH, '//p[@class="idxkeyword"]/a')

# Initialize empty lists to store href and text values
href_list = []
text_list = []

# Initialize an empty dictionary to store href and text values
static_dict = {}
for a_element in a_elements:
    href = a_element.get_attribute('href')
    text = a_element.text

    # Check if the href is not equal to 'javascript:void(0)'
    if href != 'javascript:void(0)':
        static_dict[text] = href

# Print the dictionary
print(static_dict)

# Create one dictionary with all keyword and link pairs
keyword_link_dict ={}
keyword_link_dict = {**dynamic_dict, **static_dict}
sorted_dict = {key: keyword_link_dict[key] for key in sorted(keyword_link_dict)}
print(sorted_dict)

file_path = intent_less_path + r'\ETL_web_help_articles\keyword-link-pairs.txt'

# Open the file for writing
with open(file_path, 'w') as file:
    # Iterate through the dictionary and write each key-value pair with a newline character
    for key, value in sorted_dict.items():
        file.write(f"'{key}': '{value}'\n")


# 3. Scrape content of pages stored in keyword-link-pairs.txt-----------------------------------------------------------

# Specify the directory where the ChromeDriver executable is located
chromedriver_path = r'C:\Program Files (x86)\chromedriver.exe'

# Initialize the ChromeDriver service
chrome_service = Service(executable_path=chromedriver_path)

# Initialize the WebDriver with headless option
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=chrome_service, options=options)

# Scrape Links in dict
save_directory = intent_less_path + r'\data\scraped_web_support_files'
for key, url in sorted_dict.items():
    # Specify the full path to save the file
    print(url)
    driver.get(url)

    # Wait for the iframe to load
    iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'hmcontent')))

    # Switch to the iframe
    driver.switch_to.frame(iframe)

    # Find and print the scraped text
    try:
        navigation = driver.find_elements(By.XPATH, '//p[@class="crumbs"]/a')

        breadcrumb = ""
        # Extract and print the href attributes
        for a_element in navigation:
            link = a_element.get_attribute('href')
            link_word = link.split('/hm_')[1].split('.htm')[0]
            breadcrumb += link_word + '>'
        breadcrumb = breadcrumb[:-1]
        print(breadcrumb)

        scraped_elements = driver.find_elements(By.CSS_SELECTOR, '.p_Normal')

        # Get the text content
        content = '\n'.join([element.text for element in scraped_elements])
        print(content)

        # Extract the filename from the URL
        filename = url.split('webhelp/hm_')[1] + '.txt'

        file_path = os.path.join(save_directory, filename)

        # Save content to the specified file path
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f'Key: {key}\n')
            f.write(breadcrumb + '\n')
            f.write(content + '\n')
            f.write(f'\nLink: {url}\n\n')

    except Exception as e:
        print('Error:', e)

    # Switch back to the default content (outside the iframe)
    driver.switch_to.default_content()

    # Introduce a delay between requests to avoid overloading the website
    time.sleep(5)