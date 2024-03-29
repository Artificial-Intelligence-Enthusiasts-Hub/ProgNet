# ProgNet

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

ProgNet - это языковая модель, обученная на репозиториях GitHub, которая может помочь программистам писать код, исправлять ошибки и улучшать качество своих проектов.
 
## Особенности
 
- ProgNet поддерживает несколько языков программирования, таких как Python, Java, C#, JavaScript и другие.
- ProgNet может генерировать код по естественному языку, комментариям или частичному коду, используя технику кодирования-декодирования с вниманием.
- ProgNet может анализировать код на наличие синтаксических и логических ошибок, а также предлагать возможные исправления или улучшения.
- ProgNet может адаптироваться к стилю и соглашениям кодирования конкретного проекта или пользователя, используя технику обучения с переносом.

## Техническое задание для проекта "Языковая модель для помощи программистам в Telegram"

### Цель проекта
Разработка языковой модели, предоставляющей решения и советы по кодированию в конкретных языках программирования и интеграция её в Telegram для взаимодействия в реальном времени.
### Основные требования
- Модель должна быть интегрирована с Telegram и использовать преобработчик GigaChain для анализа запросов.
- Модель должна быть способна обрабатывать и предоставлять решения в рамках установленного времени в 1-2 секунды.
- Модель должна поддерживать многопользовательское взаимодействие, обслуживая до 5000 пользователей одновременно.
### Функциональные требования
1. Понимание и обработка программного кода для приведенных выше языков программирования.
2. Предоставление конкретных решений и советов по устранению типичных ошибок програмирования.
3. Обучение на основе анализа предыдущих запросов и взаимодействий с пользователями для повышения точности ответов.
### Нефункциональные требования
1. Скорость реакции не более 1-2 секунд после получения запроса от пользователя.
2. Масштабируемость для обслуживания до 5000 пользователей одновременно без ущерба для скорости и качества.
### Этапы разработки
1. Определение и анализ конкретных потребностей программистов.
2. Проектирование системы с учетом спецификации взаимодействия и обучения модели.
3. Разработка и тестирование модуля для каждого языка программирования отдельно.
4. Тестирование интеграции модуля с Telegram и GigaChain.
5. Испытание и оптимизация масштабируемости системы.
6. Официальный релиз и постоянное обновление модели.
### Сроки проекта
Сроки проекта будут определены после детализации этапов разработки и тестирования.
### Бюджет проекта
Бюджет будет скорректирован с учетом дополнительных требований и этапов.
### Критерии приемки
- Модель успешно работает с заданными языками программирования в Telegram.
- Модель эффективно использует преобработчик GigaChain.
- Модель демонстрирует высокую скорость обработки запросов.
- Модель обеспечивает точное и конкретное решение программных проблем.
### Риски проекта
- Возможные задержки из-за технических сложностей интеграции или необходимости углубленного обучения модели.
- Возможность недооценки требований к масштабируемости и производительности.
- Риск недостаточного качества предлагаемых решений и советов.
### Предполагаемые результаты
- Увеличение производительности программистов за счет быстрого доступа к решениям и советам.
- Укрепление доверия пользователей к модели благодаря точности предоставляемых данных.
- Повышение качества кода и снижение времени на дебаггинг.

## Пользовательские инструкции

### Начало работы
- Для использования сервиса, найдите бота в Telegram по имени.
- Начните диалог с ботом, отправив команду `/start`.

### Основные команды
- `/help` - получение списка доступных команд и их описаний.
- `/ask [ваш вопрос или код]` - отправка запроса для анализа кода или получения помощи по конкретному вопросу программирования.
- `/language [язык программирования]` - настройка предпочтительного языка программирования для вашего запроса (например, Java, Python, C++).
- `/feedback` - отправка отзыва или предложения по улучшению сервиса.

### Отправка кода
- Для лучшего анализа кода, отправляйте его внутри блока кода, используя соответствующее оформление Telegram (обратные кавычки и т. д.).
- Указывайте, с какой проблемой вы столкнулись и что ожидаете от бота.

### Получение решений
- После обработки запроса бот предоставит конкретное решение или совет.
- Если необходимо уточнение, бот попросит дополнительную информацию.

### Взаимодействие с ботом
- Точно описывайте свою проблему или задачу. Соблюдайте нормы этики.
- При возникновении проблем с ботом используйте команду `/support` для связи с технической поддержкой.

### Обучение модели
- Пользователи могут улучшать модель, предоставляя обратную связь о качестве ответов и предложений.
- Задействуйте функцию `/improve` для отправки конкретных рекомендаций по улучшению модели.

> Соблюдение этих инструкций поможет боту более эффективно понимать и решать задачи, а пользователям — получать качественную помощь.
