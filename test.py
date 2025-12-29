import subprocess
import time
import os 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -------- PATH CONFIGURATION --------
APP_DIR = r"D:\mega 2\Pisorisis detection"
APP_FILE = "app1.py"
UPLOAD_FILE = r"D:\mega 2\Pisorisis detection\DATAP\PSORIASIS\08OilSpotPsoriasis1027.jpg"
URL = "http://127.0.0.1:5000"

# -------- STEP 1: RUN FLASK APP --------
print("🚀 Starting Flask app...")
process = subprocess.Popen(["python", APP_FILE], cwd=APP_DIR)

# Give Flask time to start
time.sleep(10)

# -------- STEP 2: SETUP SELENIUM --------
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)  # Keep browser open
service = Service()  # Use default ChromeDriver

driver = webdriver.Chrome(service=service, options=chrome_options)
driver.maximize_window()

# -------- STEP 3: OPEN URL --------
print("🌐 Opening the web app...")
driver.get(URL)

# -------- STEP 4: WAIT FOR PAGE TO LOAD --------
wait = WebDriverWait(driver, 10)
file_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))

# -------- STEP 5: UPLOAD IMAGE FILE --------
print("📁 Uploading image file...")
file_input.send_keys(UPLOAD_FILE)

# -------- STEP 6: CLICK 'PREDICT' BUTTON --------
predict_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Predict')]")))
predict_button.click()

print("✅ Prediction process started!")





time.sleep(5)
try:
    # Get prediction and confidence from <p> tags
    prediction_text = driver.find_element(By.XPATH, "//p[strong[contains(text(),'Prediction:')]]").text
    confidence_text = driver.find_element(By.XPATH, "//p[strong[contains(text(),'Model Confidence:')]]").text

    print("🧠", prediction_text)
    print("📊", confidence_text)

    # Optional: Extract AI-generated report
    report_text = driver.find_element(By.XPATH, "//div[@class='report']/p").text
    print("🧾 AI Report:", report_text)

except Exception as e:
    print("⚠️ Could not locate one or more result elements:", e)




    

# -------- (OPTIONAL) CLOSE EVERYTHING --------
process.terminate()
driver.quit()
