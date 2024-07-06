from selenium import webdriver

# Create a WebDriver instance
driver = webdriver.Chrome('/path/to/chromedriver')

# Open Amazon website
driver.get('https://www.amazon.com')

# Example Test: Verify Title
assert "Amazon" in driver.title, "Amazon title doesn't match"

# Example Test: Search for a Product
search_box = driver.find_element_by_id('twotabsearchtextbox')
search_box.send_keys('Python programming books')
search_box.submit()

# Example Test: Verify Search Results
assert "Python programming books" in driver.title, "Search results page title doesn't match"

# Example Test: Click on the First Result
first_result = driver.find_element_by_css_selector('.s-search-results .s-result-item:nth-child(1) h2 a')
first_result.click()

# Example Test: Verify Product Page
assert "Python programming books" in driver.title, "Product page title doesn't match"

# Close the browser
driver.quit()
