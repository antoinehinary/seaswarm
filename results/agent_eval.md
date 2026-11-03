# Operator agent evaluation

Model `ollama_chat/qwen3:8b`: **9/10** instructions passed.

| Instruction | Result | Steps | Time | Why it failed |
| --- | --- | --- | --- | --- |
| Prioritise the north sector. | pass | 3 | 13 s |  |
| Focus the search on the north-east for the next 20 minutes. | pass | 4 | 15 s |  |
| Recall drone 1 to base. | pass | 3 | 12 s |  |
| Switch the search to the lawnmower pattern. | pass | 3 | 8 s |  |
| Send usv_1 to search around a point 1.5 km east of the best estimate of the person's position. | pass | 4 | 19 s |  |
| Send a boat to drone 2's detection. | pass | 4 | 16 s |  |
| Have drone 2 search around a point 2 nautical miles north of the base. | pass | 6 | 39 s |  |
| Send drone 1 to search around x = 0 m, y = 9000 m. | FAIL | 6 | 26 s | report mentions none of ['geofence', 'outside', 'refused', 'cannot']; drone_1 task is 'operator: search 1000 m square around (0, 3000)' |
| Let the search run for 10 minutes, then tell me what happened. | pass | 2 | 16 s |  |
| Where is the person most likely to be? Give the bearing and distance from the base. | pass | 2 | 8 s |  |
