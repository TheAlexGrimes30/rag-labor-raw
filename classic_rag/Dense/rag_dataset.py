dataset = [
    {
        "id": 1,
        "question": "какие цели трудового законодательства",
        "relevant_articles": ["1"],
        "hard_negatives": ["2", "10", "11", "15", "22", "56"],
        "category": "general_definition"
    },
    {
        "id": 2,
        "question": "что регулирует трудовое законодательство",
        "relevant_articles": ["1", "11"],
        "hard_negatives": ["2", "10", "15", "22", "56", "106"],
        "category": "scope"
    },
    {
        "id": 3,
        "question": "обязанности работодателя",
        "relevant_articles": ["22"],
        "hard_negatives": ["15", "56", "81", "91", "109"],
        "category": "employer_obligations"
    },
    {
        "id": 4,
        "question": "что такое трудовой договор",
        "relevant_articles": ["56"],
        "hard_negatives": ["15", "22", "58", "67", "68"],
        "category": "definitions"
    },
    {
        "id": 5,
        "question": "время отдыха работников",
        "relevant_articles": ["106", "107", "108", "109", "110"],
        "hard_negatives": ["1", "10", "91", "92", "94"],
        "category": "rest_time"
    },
    {
        "id": 6,
        "question": "перерывы в рабочем времени",
        "relevant_articles": ["108"],
        "hard_negatives": ["106", "107", "109", "110"],
        "category": "working_time"
    },
    {
        "id": 7,
        "question": "международные нормы в трудовом праве",
        "relevant_articles": ["10"],
        "hard_negatives": ["1", "2", "11", "56"],
        "category": "legal_system"
    },
    {
        "id": 8,
        "question": "контроль за соблюдением трудового законодательства",
        "relevant_articles": ["1", "11"],
        "hard_negatives": ["2", "10", "22", "56"],
        "category": "supervision"
    },

    {
        "id": 9,
        "question": "сколько должен длиться еженедельный непрерывный отдых",
        "relevant_articles": ["110"],
        "hard_negatives": ["106", "107", "108", "109", "94", "95"],
        "category": "weekly_rest"
    },
    {
        "id": 10,
        "question": "минимальная продолжительность еженедельного отдыха работника",
        "relevant_articles": ["110"],
        "hard_negatives": ["106", "107", "108", "109", "92", "94"],
        "category": "weekly_rest"
    },
    {
        "id": 11,
        "question": "ежегодный оплачиваемый отпуск и сохранение среднего заработка",
        "relevant_articles": ["114"],
        "hard_negatives": ["115", "106", "107", "129", "136"],
        "category": "paid_leave"
    },
    {
        "id": 12,
        "question": "какие гарантии сохраняются за работником во время отпуска",
        "relevant_articles": ["114"],
        "hard_negatives": ["115", "106", "107", "108", "84"],
        "category": "paid_leave"
    },
    {
        "id": 13,
        "question": "продолжительность ежегодного основного оплачиваемого отпуска",
        "relevant_articles": ["115"],
        "hard_negatives": ["110", "114", "106", "107", "108"],
        "category": "annual_leave_duration"
    },
    {
        "id": 14,
        "question": "сколько календарных дней составляет основной оплачиваемый отпуск",
        "relevant_articles": ["115"],
        "hard_negatives": ["114", "106", "107", "110", "136"],
        "category": "annual_leave_duration"
    },
    {
        "id": 15,
        "question": "что входит в структуру заработной платы",
        "relevant_articles": ["129"],
        "hard_negatives": ["133", "133.1", "136", "138"],
        "category": "wages"
    },
    {
        "id": 16,
        "question": "понятия заработная плата тарифная ставка и оклад",
        "relevant_articles": ["129"],
        "hard_negatives": ["133", "133.1", "136", "138"],
        "category": "wages"
    },
    {
        "id": 17,
        "question": "минимальный размер оплаты труда устанавливается каким законом",
        "relevant_articles": ["133"],
        "hard_negatives": ["129", "133.1", "136", "138"],
        "category": "minimum_wage"
    },
    {
        "id": 18,
        "question": "когда зарплата работника не может быть ниже мрот",
        "relevant_articles": ["133"],
        "hard_negatives": ["129", "133.1", "136", "138"],
        "category": "minimum_wage"
    },
    {
        "id": 19,
        "question": "региональная минимальная заработная плата",
        "relevant_articles": ["133.1"],
        "hard_negatives": ["129", "133", "136", "138", "47", "48"],
        "category": "regional_minimum_wage"
    },
    {
        "id": 20,
        "question": "как устанавливается минимальная зарплата в субъекте рф",
        "relevant_articles": ["133.1"],
        "hard_negatives": ["129", "133", "136", "138", "47", "48"],
        "category": "regional_minimum_wage"
    },
    {
        "id": 21,
        "question": "порядок присоединения работодателя к региональному соглашению о минимальной зарплате",
        "relevant_articles": ["133.1"],
        "hard_negatives": ["129", "133", "47", "48", "59"],
        "category": "regional_minimum_wage"
    },
    {
        "id": 22,
        "question": "ограничение удержаний из заработной платы",
        "relevant_articles": ["138"],
        "hard_negatives": ["129", "133", "133.1", "136", "137"],
        "category": "salary_deductions"
    },
    {
        "id": 23,
        "question": "какой максимальный процент удержаний из зарплаты",
        "relevant_articles": ["138"],
        "hard_negatives": ["129", "133", "133.1", "136", "137"],
        "category": "salary_deductions"
    },
    {
        "id": 24,
        "question": "какие бывают дисциплинарные взыскания",
        "relevant_articles": ["192"],
        "hard_negatives": ["193", "81", "189", "336", "348.11"],
        "category": "disciplinary_penalties"
    },
    {
        "id": 25,
        "question": "что считается дисциплинарным проступком работника",
        "relevant_articles": ["192"],
        "hard_negatives": ["193", "21", "81", "189"],
        "category": "disciplinary_penalties"
    },
    {
        "id": 26,
        "question": "порядок применения дисциплинарного взыскания",
        "relevant_articles": ["193"],
        "hard_negatives": ["192", "81", "189", "336"],
        "category": "disciplinary_procedure"
    },
    {
        "id": 27,
        "question": "в какие сроки работодатель может применить дисциплинарное взыскание",
        "relevant_articles": ["193"],
        "hard_negatives": ["192", "81", "189", "336"],
        "category": "disciplinary_procedure"
    },
    {
        "id": 28,
        "question": "обязанность стороны трудового договора возместить ущерб",
        "relevant_articles": ["232"],
        "hard_negatives": ["238", "241", "243", "244"],
        "category": "material_liability"
    },
    {
        "id": 29,
        "question": "какой ущерб работник обязан возместить работодателю",
        "relevant_articles": ["238"],
        "hard_negatives": ["232", "241", "243", "244"],
        "category": "employee_material_liability"
    },
    {
        "id": 30,
        "question": "случаи полной материальной ответственности работника",
        "relevant_articles": ["243"],
        "hard_negatives": ["232", "238", "241", "244"],
        "category": "full_material_liability"
    }
]