import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options

# Настройка опций для Edge
options = Options()
options.add_argument("start-maximized")  # Запуск в полном экраном режиме
options.add_argument("disable-infobars")  # Отключение информационных панелей
options.add_argument("--disable-extensions")  # Отключение расширений
options.add_argument("--disable-gpu")  # Отключение GPU (ускорение графики)
options.add_argument("--no-sandbox")  # Без песочницы
options.add_argument("--disable-dev-shm-usage")  # Отключение использования /dev/shm
options.add_argument("--disable-blink-features=AutomationControlled")  # Отключение автоматического управления
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # Установка пользовательского агента

# Инициализация веб-драйвера
driver = webdriver.Edge(options=options)

try:
    # Открыть указанный сайт
    driver.get("https://vsesdal.com")

    # Можно добавить дополнительные действия здесь, если необходимо
    time.sleep(4)

finally:
    # Закрыть браузер
    driver.quit()
