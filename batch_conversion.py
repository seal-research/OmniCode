import subprocess

# Your dictionary
pr_dict = {
    "apache_dubbo": [10638, 11781, 7041],
    "elastic_logstash": [
        17021, 17020, 17019, 16968, 16681, 16579, 16569, 16482, 16195, 16094, 16079,
        15969, 15964, 15928, 15925, 15697, 15680, 15241, 15233, 15008, 15000,
        14981, 14970, 14898, 14897, 14878, 14571, 14058, 14045, 14027, 14000,
        13997, 13931, 13930, 13914, 13902, 13880, 13825
    ],
    "alibaba_fastjson2": [2775, 2559, 2285, 2097, 1245, 82],
    "fasterxml_jackson-core": [
        1309, 1263, 1208, 1204, 1182, 1172, 1142, 1053, 1016, 964,
        922, 891, 729, 566, 370, 183, 174, 980
    ],
    "fasterxml_jackson-dataformat-xml": [638, 590, 544, 531],
    "fasterxml_jackson-databind": [
        4641, 4615, 4487, 4486, 4469, 4468, 4426, 4365, 4360, 4338,
        4325, 4320, 4311, 4304, 4257, 4230, 4228, 4219, 4189, 4186,
        4159, 4132, 4131, 4087, 4072, 4050, 4048, 4015, 4013, 3860,
        3716, 3701, 3666, 3626, 3625, 3621, 3560, 3509, 3371, 2036,
        1923, 3851
    ],
    "google_gson": [1787, 1703, 1555, 1391, 1093],
    "googlecontainertools_jib": [4144, 4035, 2542, 2536, 2688],
    "mockito_mockito": [3424, 3220, 3173, 3167, 3133, 3129],
    "google_guava": [6586, 2945],
    "spring-projects_spring-boot": [
        45267, 45251, 44964, 44962, 44954, 44952, 44499, 44187, 43330,
        42856, 42852, 42174, 42067, 41213
    ]
}

# Print count for each key
counts = {key: len(value) for key, value in pr_dict.items()}
print(counts)

# Loop over each repo and each PR number
for org_repo, pr_numbers in pr_dict.items():
    org, repo = org_repo.split("_", 1)

    for pr_number in pr_numbers:
        output_file = f"sweagent_input_{org}_{repo}_{pr_number}.json"

        cmd = [
            "python", "baselines/sweagent/convert_filtered_style_errors_to_sweagent_from_dataset.py",
            "--org", org,
            "--repo", repo,
            "--pr_number", str(pr_number),
            "--style_tool", "pmd",
            "--output", output_file
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
