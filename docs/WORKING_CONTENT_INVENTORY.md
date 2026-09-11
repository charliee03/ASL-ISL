# Offline working-content inventory

Status: 11 September 2026. “Working” below means the offline route has an
available local asset and passes the project’s automated checks. It does **not**
mean universal ASL recognition accuracy, signer-certified ISL, or visual
acceptance on every browser/device.

## Approved exact-sentence translations and videos

| Item | Count | Offline behavior | Complete list |
| --- | ---: | --- | --- |
| Approved sentence inputs | 97 | Exact normalized input uses the user-approved translation rather than Gemini or generic token rules. | [Sentence translation review](OFFLINE_SENTENCE_TRANSLATION_REVIEW.csv) |
| Recorded sentence-pose videos | 663 clips for 97 unique sentence labels | The renderer selects the best available clip for an exact sentence match. | [Sentence video manifest](../data/isl/cslrt_keypoints/metadata.json) |

The review sheet contains every supported input sentence, its approved
translation, and `approved` status. The video manifest lists every corresponding
local landmark-video file and signer variant.

### Full supported sentence-input list (97)

1. are you free today
2. are you hiding something
3. bring water for me
4. can i help you
5. can you repeat that please
6. comb your hair
7. congratulations
8. could you please talk slower
9. do me a favour
10. do not abuse him
11. do not be stubborn
12. do not hurt me
13. do not make me angry
14. do not take it to the heart
15. do not worry
16. do you need something
17. go and sleep
18. had your food
19. he came by train
20. He is going into the room
21. he is on the way
22. he she is my friend
23. help me
24. hi how are you
25. how are things
26. how can i help you
27. how can i trust you
28. how dare you
29. how old are you
30. i am (age)
31. i am afraid of that
32. i am crying
33. i am feeling bored
34. i am feeling cold
35. i am fine. thank you sir
36. i am hungry
37. i am in dilemma what to do
38. i am not really sure
39. i am really grateful
40. i am sitting in the class
41. i am so sorry to hear that
42. i am suffering from fever
43. i am tired
44. i am very happy
45. i can not help you there
46. i do not agree
47. i do not like it
48. i do not mean it
49. i enjoyed a lot
50. i got hurt
51. i like you i love you
52. i need water
53. i promise
54. i really appreciate it
55. i somehow got to know about it
56. i was stopped by some one
57. it does not make any difference to me
58. it was nice chatting with you
59. let him take time
60. my name is xxxxxxxx
61. nice to meet you
62. pour some more water into the glass
63. prepare the bed
64. serve the food
65. shall we go outside
66. speak softly
67. take care of yourself
68. tell me truth
69. thank you so much
70. that is so kind of you
71. This place is beautiful
72. try to understand
73. turn on light turn off light
74. we are all with you
75. wear the shirt
76. what are you doing
77. what did you tell him
78. what do you do
79. what do you think
80. what do you want to become
81. what happened
82. what have you planned for your career
83. what is your phone number
84. what you want
85. when will the train leave
86. where are you from
87. which collegeschool are you from
88. who are you
89. why are you angry
90. why are you crying
91. why are you disappointed
92. you are bad
93. you are good
94. you are welcome
95. you can do it
96. you do anything, i do not care
97. you need a medicine, take this one

## Recorded word-pose videos

There are 3,630 local clips across these 61 playback labels. A matching label or
configured alias plays a recorded pose sequence; a multiword sequence composes
each available word and fingerspells unsupported tokens with visible gaps.

`bear`, `break`, `brinjal`, `budget`, `busy`, `cabbage`, `carrot`,
`cauliflower`, `chilli`, `clean`, `close`, `come`, `cook`, `crocodile`, `cry`,
`cucumber`, `deer`, `drink`, `elephant`, `exam`, `fedup`, `fever`, `giraffe`,
`give`, `good_afternoon`, `good_morning`, `hello`, `hug`, `injury`,
`interview`, `jump`, `karnataka`, `key`, `knife`, `lemon`, `lion`, `man`,
`maths`, `maybe`, `monkey`, `onion`, `peacock`, `pigeon`, `pour`, `radish`,
`sparrow`, `still`, `switch`, `tea`, `temple`, `thank_you`, `tiger`, `turtle`,
`umbrella`, `uncle`, `vegetables`, `volcano`, `what_is_your_name`, `wife`,
`writer`, `wrong`.

Every clip path and its word label is in the [word video manifest](../data/isl/keypoints/metadata.json).

## Packaged ASL upload demos

These three MSASL clips are the supported browser-upload demos. Each has a
saved validated feature sequence and was checked through recognition,
translation and recorded-avatar playback on 11 September 2026.

| Expected recognition | Upload file | Avatar playback |
| --- | --- | --- |
| `hello` | [class_0_signer_26_23348.mp4](../Dataset/MS-ASL/videos/class_0_signer_26_23348.mp4) | Recorded `hello` via `NAMASKAR` alias |
| `drink` | [class_56_signer_10_22073.mp4](../Dataset/MS-ASL/videos/class_56_signer_10_22073.mp4) | Recorded `drink` |
| `man` | [class_58_signer_10_24288.mp4](../Dataset/MS-ASL/videos/class_58_signer_10_24288.mp4) | Recorded `man` |

The server verifies the uploaded file hash and uses the associated saved
landmarks. Other uploads are rejected in this memory-limited local demo rather
than risking a failed extraction process.

## Alphabet fallback

35 local alphabet assets provide A–Z coverage, including alternate forms for
C, E, I, J, T, U, V and Z. Unknown words can be fingerspelled; the fallback is
technically tested but still needs sign-legibility review where it is used.

## Not classified as “working properly” for broad claims

- The MSASL recognizer has 100 isolated-sign labels, but its saved overall test
  Top-1 is 47.02%, so it is intentionally disabled in the current demo.
- Gemini is disabled; it is not needed for the 97 approved exact sentences.
- General, unseen sentences use the limited rule fallback and are not covered by
  this inventory.
