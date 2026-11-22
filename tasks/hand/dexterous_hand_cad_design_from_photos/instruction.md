# Recover the CAD of a four-digit robot hand

A lab built a four-digit robot hand from an open design: three fingers and a thumb on a palm, moved by small servo
actuators. The lab wants a second unit and a simulation model, and the CAD of the printed parts is lost: only the files
of the bought parts are left. A colleague photographed the unit on the bench, took it apart stage by stage while
photographing every stage, laid every printed part on a cutting mat next to its tag and scale references, turned each
part onto several of its sides, and took close-ups. One finger was taken apart down to its last part; the three fingers
are alike, and the other two were left assembled. The unit is in pieces in a box at another site now: nobody can measure
it again. You are the mechanical engineer who re-creates the hand in FreeCAD from those photographs and the documents of
the bought parts, so that it can be printed again and simulated.

The hand has 52 made parts in 27 different shapes (the printed parts) and 53 given parts of 8 types (the bought parts).
Infer from the evidence the shape and the dimensions of every made part, where every made and given part sits, and how
the hand moves. The reference is the unit as it stands in the photographs, not a corrected version of it; only the
grader has it.

## How you work

Use the FreeCAD MCP server `freecad` to inspect and edit the running FreeCAD 1.1 session. Its `execute_code` tool runs
Python inside that session, with FreeCAD's own modules. Your other tools remain available for looking at the photos,
measuring in images, calculations and scripts. In the shell, the Python with the scientific packages is
`/opt/venv/bin/python3`, and the headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`. There is no network.

A viewer shows a photo scaled down: to see a detail at full resolution, crop it to a file first and look at the crop.
Nothing in the environment evaluates a design against the reference. Whether you work through the MCP server or the
shell does not affect your score; the saved document does.

## Given files (in `/home/user/`)

`photos/photo_001.jpg` to `photos/photo_363.jpg`: the photo set, 363 JPEG files of 2016 x 1512 pixels. Teardown photos
show the made parts on the cutting mat, each next to its tag, a strip of label tape. A part was photographed resting on
several of its sides; the caption letters the resting position, and the same letter always means the same side of that
part. Parts with the same shape were laid out together, each with its own tag; they rest on different sides as far as
the shape has steady sides, and where it has fewer than there are pieces, some pieces rest alike. Photos with the same
tags and the same arrangement letter show one layout from different viewpoints. The tags run from P01 to P52. Close-ups
show one tagged part at close range, in the resting position its caption names. The photos of the assembled hand show it
lying on the cutting mat. A caption that says "at rest" marks the rest pose; every other photo of the assembled hand
shows a different pose. The stage photos show the hand as it was taken apart. Each caption names what was taken off in
that stage, made parts by tag and given parts by name; everything taken off in an earlier stage stays off. What came off
in one piece in an earlier stage can lie in a later stage's photos again, taken apart, without being named a second
time. One of the three fingers was taken apart down to its last part; the three are alike, and the other two were left
assembled. A made part that no stage caption names was never taken off in a stage; the layout photos and the close-ups
show it loose all the same. Two photos show the cutting mat's grid alone. The camera is perspective and uncalibrated,
and file names do not encode its pose. All photos come from one phone with three lenses whose focal lengths are in the
ratio 1 : 2 : 5; no caption says which lens took a photo. Exposure and white balance vary a little.

| photos | what they show | scale references |
|---|---|---|
| `photo_001`-`002` | P01, resting position A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_003`-`004` | P01, resting position B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_005` | P01, resting position C | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_006` | P01, resting position C | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_007`-`008` | P02, resting position A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_009` | P02, resting position B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_010` | P02, resting position B | cutting mat, small checker card (partly in frame), steel rule |
| `photo_011`-`012` | P02, resting position C | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_013` | P03, resting position A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_014` | P03, resting position A | cutting mat, small checker card, steel rule |
| `photo_015` | P03, resting position B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_016` | P03, resting position B | cutting mat, small checker card, steel rule |
| `photo_017` | P03, resting position C | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_018` | P03, resting position C | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_019` | P04, resting position A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_020` | P04, resting position A | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_021` | P04, resting position A | cutting mat, small checker card, steel rule |
| `photo_022`-`023` | P04, resting position B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_024` | P04, resting position B | cutting mat, small checker card, steel rule |
| `photo_025` | P05, resting position A | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_026` | P05, resting position A | cutting mat, small checker card, steel rule |
| `photo_027` | P05, resting position B | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_028` | P05, resting position B | cutting mat, small checker card, steel rule |
| `photo_029` | P05, resting position C | cutting mat, checker card (partly in frame), small checker card, steel rule |
| `photo_030` | P05, resting position C | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_031`-`032` | P16, resting position A | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_033`-`034` | P16, resting position B | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_035` | P16, resting position C | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_036` | P16, resting position C | cutting mat, checker card (partly in frame), small checker card, steel rule (partly in frame) |
| `photo_037` | P06, P33-P36, P46, arrangement A | cutting mat, small checker card |
| `photo_038` | P06, P33-P36, P46, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_039` | P06, P33-P36, P46, arrangement A | cutting mat, small checker card |
| `photo_040` | P06, P33-P36, P46, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_041`-`042` | P06, P33-P36, P46, arrangement A | cutting mat, steel rule (partly in frame) |
| `photo_043`-`044` | P06, P33-P36, P46, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_045` | P06, P33-P36, P46, arrangement B | cutting mat, small checker card |
| `photo_046` | P06, P33-P36, P46, arrangement B | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_047` | P06, P33-P36, P46, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_048`-`049` | P07, P29-P31, P41, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_050` | P07, P29-P31, P41, arrangement A | cutting mat, small checker card |
| `photo_051` | P07, P29-P31, P41, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_052` | P07, P29-P31, P41, arrangement B | cutting mat, small checker card |
| `photo_053`-`054` | P07, P29-P31, P41, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_055` | P08, P17-P19, arrangement A | cutting mat, small checker card |
| `photo_056` | P08, P17-P19, arrangement A | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_057` | P08, P17-P19, arrangement A | cutting mat, small checker card |
| `photo_058`-`059` | P08, P17-P19, arrangement A | cutting mat, steel rule (partly in frame) |
| `photo_060`-`062` | P08, P17-P19, arrangement B | cutting mat, small checker card |
| `photo_063`-`064` | P08, P17-P19, arrangement B | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_065`-`067` | P09, P20-P22, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_068`-`070` | P09, P20-P22, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_071`-`072` | P09, P20-P22, arrangement B | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_073` | P10-P12, arrangement A | cutting mat, small checker card |
| `photo_074` | P10-P12, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_075` | P10-P12, arrangement A | cutting mat, small checker card |
| `photo_076` | P10-P12, arrangement A | cutting mat, steel rule (partly in frame) |
| `photo_077`-`079` | P10-P12, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_080`-`081` | P13-P15, P48, P49, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_082` | P13-P15, P48, P49, arrangement A | cutting mat, small checker card |
| `photo_083`-`084` | P13-P15, P48, P49, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_085` | P13-P15, P48, P49, arrangement B | cutting mat, small checker card |
| `photo_086` | P13-P15, P48, P49, arrangement B | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_087`-`091` | P23-P25, P32, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_092` | P23-P25, P32, arrangement A | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_093` | P23-P25, P32, arrangement B | cutting mat, small checker card |
| `photo_094` | P23-P25, P32, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_095` | P23-P25, P32, arrangement B | cutting mat, small checker card |
| `photo_096`-`098` | P26-P28, P50-P52, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_099` | P26-P28, P50-P52, arrangement B | cutting mat, small checker card |
| `photo_100` | P26-P28, P50-P52, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_101` | P26-P28, P50-P52, arrangement B | cutting mat, small checker card |
| `photo_102` | P37-P40, P42-P45, P47, arrangement A | cutting mat, small checker card |
| `photo_103` | P37-P40, P42-P45, P47, arrangement A | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_104` | P37-P40, P42-P45, P47, arrangement A | cutting mat, small checker card |
| `photo_105` | P37-P40, P42-P45, P47, arrangement A | cutting mat, steel rule (partly in frame) |
| `photo_106` | P37-P40, P42-P45, P47, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_107` | P37-P40, P42-P45, P47, arrangement B | cutting mat, small checker card (partly in frame) |
| `photo_108` | P37-P40, P42-P45, P47, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_109` | P37-P40, P42-P45, P47, arrangement B | cutting mat, small checker card (partly in frame), steel rule (partly in frame) |
| `photo_110` | P37-P40, P42-P45, P47, arrangement B | cutting mat, small checker card, steel rule (partly in frame) |
| `photo_111` | all made parts with their tags | cutting mat, checker card (partly in frame), steel rule (partly in frame) |
| `photo_112` | all made parts with their tags | cutting mat, checker card, steel rule |
| `photo_113` | all made parts with their tags | cutting mat, checker card (partly in frame), steel rule (partly in frame) |
| `photo_114` | all made parts with their tags | cutting mat, steel rule |
| `photo_115` | all made parts with their tags | cutting mat, checker card, steel rule (partly in frame) |
| `photo_116` | all made parts with their tags | cutting mat, checker card (partly in frame), steel rule (partly in frame) |
| `photo_117` | all made parts with their tags | cutting mat, checker card, steel rule (partly in frame) |
| `photo_118` | all made parts with their tags | cutting mat, checker card, steel rule |
| `photo_119` | close-up of P06, resting position A | cutting mat |
| `photo_120` | close-up of P06, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_121` | close-up of P06, resting position A | cutting mat |
| `photo_122` | close-up of P06, resting position B | cutting mat |
| `photo_123` | close-up of P06, resting position B | cutting mat, small checker card |
| `photo_124` | close-up of P06, resting position B | cutting mat |
| `photo_125`-`127` | close-up of P07, resting position A | cutting mat |
| `photo_128`-`129` | close-up of P07, resting position B | cutting mat |
| `photo_130` | close-up of P07, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_131`-`133` | close-up of P08, resting position A | cutting mat |
| `photo_134` | close-up of P08, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_135` | close-up of P08, resting position B | cutting mat |
| `photo_136` | close-up of P08, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_137` | close-up of P08, resting position C | cutting mat |
| `photo_138` | close-up of P08, resting position C | cutting mat, small checker card (partly in frame) |
| `photo_139` | close-up of P08, resting position C | cutting mat |
| `photo_140`-`141` | close-up of P09, resting position A | cutting mat |
| `photo_142` | close-up of P09, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_143` | close-up of P09, resting position B | cutting mat |
| `photo_144` | close-up of P09, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_145` | close-up of P09, resting position B | cutting mat |
| `photo_146`-`147` | close-up of P09, resting position C | cutting mat |
| `photo_148` | close-up of P09, resting position C | cutting mat, small checker card (partly in frame) |
| `photo_149`-`150` | close-up of P10, resting position A | cutting mat |
| `photo_151` | close-up of P10, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_152`-`153` | close-up of P10, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_154` | close-up of P10, resting position B | cutting mat |
| `photo_155` | close-up of P13, resting position A | cutting mat |
| `photo_156` | close-up of P13, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_157` | close-up of P13, resting position A | cutting mat |
| `photo_158`-`159` | close-up of P13, resting position B | cutting mat |
| `photo_160` | close-up of P13, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_161`-`163` | close-up of P13, resting position C | cutting mat |
| `photo_164` | close-up of P17, resting position A | cutting mat |
| `photo_165` | close-up of P17, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_166` | close-up of P17, resting position A | cutting mat |
| `photo_167`-`169` | close-up of P17, resting position B | cutting mat |
| `photo_170` | close-up of P17, resting position C | cutting mat |
| `photo_171`-`172` | close-up of P17, resting position C | cutting mat, small checker card (partly in frame) |
| `photo_173` | close-up of P20, resting position A | cutting mat |
| `photo_174` | close-up of P20, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_175` | close-up of P20, resting position A | cutting mat |
| `photo_176`-`178` | close-up of P20, resting position B | cutting mat |
| `photo_179`-`180` | close-up of P23, resting position A | cutting mat |
| `photo_181` | close-up of P23, resting position A | cutting mat, small checker card |
| `photo_182` | close-up of P23, resting position B | cutting mat |
| `photo_183` | close-up of P23, resting position B | cutting mat, small checker card |
| `photo_184` | close-up of P23, resting position B | cutting mat |
| `photo_185`-`187` | close-up of P26, resting position A | cutting mat |
| `photo_188` | close-up of P26, resting position B | cutting mat |
| `photo_189` | close-up of P26, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_190` | close-up of P26, resting position B | cutting mat |
| `photo_191`-`193` | close-up of P29, resting position A | cutting mat |
| `photo_194`-`195` | close-up of P29, resting position B | cutting mat |
| `photo_196` | close-up of P29, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_197` | close-up of P32, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_198` | close-up of P32, resting position A | cutting mat, small checker card |
| `photo_199` | close-up of P32, resting position A | cutting mat |
| `photo_200`-`201` | close-up of P32, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_202` | close-up of P32, resting position B | cutting mat |
| `photo_203` | close-up of P33, resting position A | cutting mat |
| `photo_204` | close-up of P33, resting position A | cutting mat, small checker card |
| `photo_205` | close-up of P33, resting position A | cutting mat |
| `photo_206`-`207` | close-up of P33, resting position B | cutting mat |
| `photo_208` | close-up of P33, resting position B | cutting mat, small checker card |
| `photo_209`-`210` | close-up of P37, resting position A | cutting mat |
| `photo_211` | close-up of P37, resting position A | cutting mat, small checker card |
| `photo_212` | close-up of P37, resting position B | cutting mat |
| `photo_213`-`214` | close-up of P37, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_215`-`216` | close-up of P41, resting position A | cutting mat |
| `photo_217` | close-up of P41, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_218`-`220` | close-up of P41, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_221` | close-up of P42, resting position A | cutting mat |
| `photo_222`-`223` | close-up of P42, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_224` | close-up of P42, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_225`-`226` | close-up of P42, resting position B | cutting mat, small checker card |
| `photo_227` | close-up of P45, resting position A | cutting mat |
| `photo_228` | close-up of P45, resting position A | cutting mat, small checker card |
| `photo_229` | close-up of P45, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_230`-`231` | close-up of P45, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_232` | close-up of P45, resting position B | cutting mat, small checker card |
| `photo_233` | close-up of P46, resting position A | cutting mat |
| `photo_234`-`235` | close-up of P46, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_236`-`237` | close-up of P46, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_238` | close-up of P46, resting position B | cutting mat, small checker card |
| `photo_239` | close-up of P47, resting position A | cutting mat |
| `photo_240` | close-up of P47, resting position A | cutting mat, small checker card |
| `photo_241` | close-up of P47, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_242` | close-up of P47, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_243` | close-up of P47, resting position B | cutting mat, small checker card |
| `photo_244` | close-up of P47, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_245` | close-up of P48, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_246`-`247` | close-up of P48, resting position A | cutting mat, small checker card |
| `photo_248` | close-up of P48, resting position B | cutting mat |
| `photo_249`-`250` | close-up of P48, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_251`-`253` | close-up of P50, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_254` | close-up of P50, resting position B | cutting mat |
| `photo_255` | close-up of P50, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_256` | close-up of P50, resting position B | cutting mat, small checker card |
| `photo_257`-`259` | close-up of P01, resting position A | cutting mat |
| `photo_260`-`261` | close-up of P01, resting position B | cutting mat |
| `photo_262`-`263` | close-up of P01, resting position C | cutting mat |
| `photo_264`-`266` | close-up of P02, resting position A | cutting mat |
| `photo_267` | close-up of P02, resting position B | cutting mat |
| `photo_268` | close-up of P02, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_269`-`270` | close-up of P02, resting position C | cutting mat |
| `photo_271`-`273` | close-up of P03, resting position A | cutting mat |
| `photo_274`-`275` | close-up of P03, resting position B | cutting mat |
| `photo_276`-`277` | close-up of P03, resting position C | cutting mat |
| `photo_278`-`280` | close-up of P04, resting position A | cutting mat |
| `photo_281` | close-up of P04, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_282` | close-up of P04, resting position B | small checker card |
| `photo_283`-`284` | close-up of P04, resting position B | cutting mat |
| `photo_285`-`287` | close-up of P05, resting position A | cutting mat |
| `photo_288`-`289` | close-up of P05, resting position B | cutting mat |
| `photo_290`-`291` | close-up of P05, resting position C | cutting mat |
| `photo_292`-`293` | close-up of P16, resting position A | cutting mat, small checker card |
| `photo_294` | close-up of P16, resting position A | cutting mat, small checker card (partly in frame) |
| `photo_295`-`296` | close-up of P16, resting position B | cutting mat, small checker card (partly in frame) |
| `photo_297`-`298` | close-up of P16, resting position C | cutting mat, small checker card (partly in frame) |
| `photo_299` | assembled hand at rest, palm down | cutting mat, steel rule |
| `photo_300` | assembled hand at rest, palm down | cutting mat, checker card (partly in frame) |
| `photo_301` | assembled hand at rest, palm down | cutting mat, checker card, steel rule |
| `photo_302` | assembled hand at rest, palm down | cutting mat, checker card (partly in frame), steel rule |
| `photo_303` | assembled hand at rest, palm down | cutting mat, steel rule |
| `photo_304` | assembled hand at rest, palm down | cutting mat, checker card, steel rule (partly in frame) |
| `photo_305`-`306` | assembled hand at rest, palm down | cutting mat, checker card, steel rule |
| `photo_307` | assembled hand at rest, palm down | cutting mat, checker card (partly in frame), steel rule |
| `photo_308` | assembled hand at rest, palm up | cutting mat, checker card |
| `photo_309` | assembled hand at rest, palm up | cutting mat, checker card, steel rule (partly in frame) |
| `photo_310` | assembled hand at rest, palm up | cutting mat, checker card (partly in frame), steel rule |
| `photo_311` | assembled hand at rest, palm up | cutting mat, steel rule |
| `photo_312` | assembled hand in pose A, which is not the rest pose | cutting mat, steel rule (partly in frame) |
| `photo_313` | assembled hand in pose A, which is not the rest pose | cutting mat, checker card |
| `photo_314` | assembled hand in pose A, which is not the rest pose | cutting mat, steel rule |
| `photo_315` | assembled hand in pose A, which is not the rest pose | cutting mat, checker card (partly in frame), steel rule (partly in frame) |
| `photo_316` | assembled hand in pose B, which is not the rest pose | cutting mat, checker card |
| `photo_317` | assembled hand in pose B, which is not the rest pose | cutting mat, checker card (partly in frame), steel rule (partly in frame) |
| `photo_318` | assembled hand in pose B, which is not the rest pose | cutting mat, checker card (partly in frame), steel rule |
| `photo_319` | assembled hand in pose B, which is not the rest pose | cutting mat, steel rule |
| `photo_320` | index finger of the assembled hand seen from the side, at rest | cutting mat, small checker card standing upright |
| `photo_321` | index finger of the assembled hand seen from the other side, at rest | cutting mat, small checker card standing upright |
| `photo_322` | index finger of the assembled hand seen from the side, flexed | cutting mat, small checker card standing upright |
| `photo_323` | middle finger of the assembled hand seen from the side, at rest | cutting mat, small checker card standing upright |
| `photo_324` | middle finger of the assembled hand seen from the other side, at rest | cutting mat, small checker card standing upright |
| `photo_325` | middle finger of the assembled hand seen from the side, flexed | cutting mat, small checker card standing upright |
| `photo_326` | ring finger of the assembled hand seen from the side, at rest | cutting mat, small checker card standing upright |
| `photo_327` | ring finger of the assembled hand seen from the other side, at rest | cutting mat, small checker card standing upright |
| `photo_328` | ring finger of the assembled hand seen from the side, flexed | cutting mat, small checker card standing upright |
| `photo_329` | thumb of the assembled hand seen from the side, at rest | cutting mat, small checker card standing upright |
| `photo_330` | thumb of the assembled hand seen from the other side, at rest | cutting mat, small checker card standing upright (partly in frame) |
| `photo_331` | thumb of the assembled hand seen from the side, flexed | cutting mat, small checker card standing upright |
| `photo_332` | stage 1: P04 removed | cutting mat, checker card, small checker card, steel rule |
| `photo_333` | stage 1: P04 removed; near view | cutting mat, small checker card (partly in frame) |
| `photo_334` | stage 1: P04 removed; near view | cutting mat, small checker card |
| `photo_335` | stage 2: P05, P50-P52 and the power and bus hub board removed | cutting mat, checker card, small checker card |
| `photo_336`-`337` | stage 2: P05, P50-P52 and the power and bus hub board removed; near view | cutting mat, small checker card |
| `photo_338` | stage 3: P02 removed | cutting mat, checker card, small checker card, steel rule |
| `photo_339` | stage 3: P02 removed; near view | cutting mat |
| `photo_340` | stage 3: P02 removed; near view | cutting mat, small checker card |
| `photo_341` | stage 4: P03, P16, P41, P47, the bus distribution board and the tactile interface board removed | cutting mat, checker card, small checker card, steel rule |
| `photo_342`-`343` | stage 4: P03, P16, P41, P47, the bus distribution board and the tactile interface board removed; near view | cutting mat, small checker card |
| `photo_344` | stage 5: the three fingers and the thumb removed, each still assembled | cutting mat, checker card, small checker card, steel rule |
| `photo_345`-`346` | stage 5: the three fingers and the thumb removed, each still assembled; near view | cutting mat, small checker card |
| `photo_347` | stage 6, one of the three fingers (the three are alike; the other two stay assembled): P44 and the fingertip tactile module removed | cutting mat, checker card, small checker card, steel rule (partly in frame) |
| `photo_348`-`349` | stage 6, one of the three fingers (the three are alike; the other two stay assembled): P44 and the fingertip tactile module removed; near view | cutting mat, small checker card |
| `photo_350` | stage 7, the same finger: P12, P15, P22, P28, P31, the miniature ball bearings and the dowel pins removed | cutting mat, small checker card, steel rule |
| `photo_351`-`353` | stage 7, the same finger: P12, P15, P22, P28, P31, the miniature ball bearings and the dowel pins removed; near view | cutting mat, small checker card |
| `photo_354` | stage 7, the same finger: P12, P15, P22, P28, P31, the miniature ball bearings and the dowel pins removed; near view | cutting mat, small checker card (partly in frame) |
| `photo_355` | stage 7, the same finger: P12, P15, P22, P28, P31, the miniature ball bearings and the dowel pins removed; near view | cutting mat, small checker card |
| `photo_356` | stage 8, the same finger: P19, P25, P35, P39 and the servo actuators removed | cutting mat, checker card, small checker card, steel rule |
| `photo_357`-`358` | stage 8, the same finger: P19, P25, P35, P39 and the servo actuators removed; near view | cutting mat, small checker card |
| `photo_359` | stage 9, the thumb: P07, P32, P36, P40, P45, P46, P48, P49, the thumb tactile module and the servo actuators removed | cutting mat, checker card, small checker card, steel rule (partly in frame) |
| `photo_360`-`361` | stage 9, the thumb: P07, P32, P36, P40, P45, P46, P48, P49, the thumb tactile module and the servo actuators removed; near view | cutting mat, small checker card |
| `photo_362`-`363` | cutting mat grid, no part in view | cutting mat |

Scale references in the photographs: cutting mat: A3, 420 x 297 mm, with a 10 mm grid printed on its top face, every
fifth line bold; checker card: ten by seven squares of 12 mm inside a white border, 132 x 96 mm overall, 1.0 mm thick;
small checker card: ten by seven squares of 2 mm inside a 3 mm white border, about 26 x 20 mm overall, 1.0 mm thick; in
the photos that show one digit of the assembled hand from the side, one stands on its long edge on the mat, beyond that
digit's fingertip or beside the hand, propped against a plain block and leaning back about 8 degrees from the vertical,
its squares turned to the side the photo is taken from; steel rule: 150 mm scale with 1 mm and 0.5 mm graduations, 0.5
mm thick.

The column "scale references" names the references that the photos of a row show. A checker card is named when at least
half of its pattern shows, the steel rule and the mat's grid when a stretch of them shows that is long enough to measure
with. On a card or the rule, the mark "(partly in frame)" stands for a reference of which the frame cuts a part off or
something in front of it hides a part. A reference of which less shows is not named, though a piece of it can be in the
frame. The references lie flat, apart from the card that stands upright: on the mat's top face, where a checker card can
reach over the mat's edge, and in one close-up, where the part fills the frame, on the part itself, above the plane of
the mat's grid. More than one small checker card was used, and they are all alike: one lies beside the tag or beside the
layout in most teardown photos, and a stage photo can show two or three of them; the column names a kind of reference
once.

`purchased/<part number>/` for each type of given part: `<part number>.step` (its geometry, one solid in the part's own
frame; the file is written in metres, and FreeCAD imports it in millimetres), `<part number>_datasheet.md` (the data
page: mass, frame, size, mounting interface) and `<part number>_drawing.png`. The sizes on a data page are the solid's
exact bounds; FreeCAD's `Shape.BoundBox` can read a little larger on a solid with free-form faces.

| part number | name | pieces | files | grading weight |
|---|---|---:|---|---:|
| `VKR-4LBA` | servo actuator | 13 | `purchased/VKR-4LBA/` | 3 |
| `HNL-EX77` | miniature ball bearing `*` | 18 | `purchased/HNL-EX77/` | 0.25 |
| `HNL-NQ37` | dowel pin `*` | 15 | `purchased/HNL-NQ37/` | 0.25 |
| `ZCP-F561` | fingertip tactile module | 3 | `purchased/ZCP-F561/` | 2 |
| `ZCP-DG1Z` | thumb tactile module | 1 | `purchased/ZCP-DG1Z/` | 2 |
| `BWD-RS8C` | bus distribution board | 1 | `purchased/BWD-RS8C/` | 1 |
| `ZCP-384F` | tactile interface board | 1 | `purchased/ZCP-384F/` | 1 |
| `JQT-NLP9` | power and bus hub board | 1 | `purchased/JQT-NLP9/` | 1 |

Grading weight: the weight of one piece wherever the grading below weighs given pieces. `*`: only the direction of the
piece's own z, either way round, is graded as its orientation; its turn about z is not.

`parts_list.md`: the made parts by tag, with pieces, material and density (PLA 1.24 g/cm³ (P01-P03, P05-P52), TPU 1.21
g/cm³ (P04)); the given parts with part number, name, pieces and mass per piece; the fasteners with type, size and
pieces. No drawing, dimension or hole table of a made part is given, and no document states which tagged part sits where
or what it does; the stage captions name only what was taken off together.

## Deliverable

Save `/home/user/submission/hand.FCStd`, one FreeCAD document, with:

1. One PartDesign Body for each tag P01 to P52, with the Body's **Label** exactly the tag and exactly one closed solid
   in it, in millimetres. Use each tag once. Tags on one line of `parts_list.md` mark parts of the same shape: give each
   its own Body; which of them carries which of those tags is not graded.
2. Every given part (53 pieces of 8 types): each piece imported unchanged from its STEP file and moved to where it is
   installed by its Placement only. Do not fuse, cut, scale or re-model a given solid: it is recognised by its shape,
   whatever its label.
3. One native FreeCAD Assembly (the Assembly workbench's object, with its joints) that contains all of these. Only what
   is inside the Assembly, directly or through containers and links, is graded.
4. Joints in that Assembly that say how the hand is built. One Grounded joint on the palm's largest made part. A Fixed
   joint for everything rigidly attached, the given parts included: a servo's case to the part that holds it, a dowel
   pin or a miniature ball bearing to either of the parts it joins. One Revolute joint for every hinge of the hand, that
   is every place where two parts turn relative to each other about an axis, between the two parts it connects, also
   where hinges close a chain of parts. Grounding a part does not connect it, and a servo that no Fixed joint holds
   drives nothing.
5. Both angle limits enabled (Min and Max angle) on every hinge whose angle is independent, at the interval through
   which that hinge turns from the saved pose while every other independent hinge stays at its saved angle: until two
   parts of the same digit, or a part of the digit and a part of the palm, meet (made and given parts count; fasteners
   and the parts of other digits do not), or until a closed chain of parts can turn no further (the limit is then 1°
   before that position). Parts that already touch at the saved pose stop the hinge there in the direction that presses
   them together. A hinge whose angle follows from others through a closed chain of parts needs no limits.
6. The Assembly saved in the rest pose, the pose of the photos whose captions say "at rest" (the whole hand:
   `photo_299`-`311`). Every hinge angle is counted from the saved pose. Origin and overall orientation are free.

Fasteners, cables and whatever else is neither a tagged part nor a given part are not part of the deliverable: do not
model them. Do not include a mat, scale references, alternative versions or duplicate solids; anything extra counts as
an extra solid (below). Keep everything in this one file: only `/home/user/submission` reaches the verifier, and a link
to an object of a document saved elsewhere is empty there.

## How it is graded

The verifier runs in a separate sandbox after your session. It opens the saved document with FreeCAD (nothing in it is
executed), exports every solid in world coordinates and reads the joints. The stored placements and joint frames are
read as they are; the Assembly is not solved again. What is graded is inside the Assembly: its members, the members of
App::Part containers, sub-assemblies and folders inside it, and links. Objects outside the Assembly are ignored, the
palm included. Of several Assemblies with components, the one that holds the palm's largest made part is read and the
others are reported. A solid counts when it is closed (`Shape.isClosed()`) and has a volume; its geometric validity is
not checked.

The whole document scores 0 in these cases only. First, it is missing or cannot be read: the submission folder holds no
`hand.FCStd` (a symbolic link in its place is not followed); FreeCAD cannot open the document, or needs more than 15
minutes to open and export it; the document has no Assembly that has components; or its Assembly holds more than 3000
solids (the complete hand has 105) or, directly and in its containers, sub-assemblies and folders, more than 4000
objects (a Body counts once; joints and datum objects do not count). Second, its Assembly holds no closed solid, or no
usable palm (below), so that nothing can be aligned. Third, the submission folder itself is not accepted: it is a
symbolic link or resolves to a place outside `/home/user/`, it holds an unchanged copy of a public source file of this
hand's CAD model, or something in it is still written after your session has ended.

Everything else is local and reported. A tag that is not on a PartDesign Body holding exactly one closed solid and
nothing else gets no credit, and neither does a tag found on more than one object; the closed solids concerned count as
extra solids. Open shells and loose faces of other objects are left out. Extra solids are all closed solids that are
neither a credited tagged part nor a given piece counted in the hand; a given piece that is in the document more often
than in the hand is extra as well. Extra solids count in the mass at 1.24 g/cm³ and are obstacles in the fit criterion
and, where they are fixed into the palm or a digit, in the interference criterion.

Given parts are recognised by their full shape, never by a label: a solid is a given piece when it is the given geometry
(volume within 1 %, then a rigid registration onto the solid of the STEP file). The pieces of one type are matched to
their installed positions one to one, and so are the tagged parts of one shape: it does not matter which of them carries
which tag. Every tagged part is first fitted onto the true part of its shape on its own, by a rigid motion, wherever it
sits in your document; its shape and its holes are read in that fit, and its placement is the pose that fit gives.

**What the photographs locate.** The grader holds a mark for every patch of the true surface of a made part, for every
hole, for every hinge and for the installed place of every made part and given piece: whether the photographs locate it,
that is, whether they show it, or the features that fix it, clearly enough and from enough directions to tell where it
is. The marks were made from the photo set alone, before any submission. Where the criteria below speak of what the
photographs show or locate, they mean these marks. What the photographs do not locate is not graded: such surface enters
neither the shape distance nor the gap of the fit criterion, such a hole has no hole item, such a hinge has no axis
item, no range item and no key length that ends at it, and such a place has no placement or installation item and no
part in the alignment; each is left out of the numerator and the denominator alike. A kinematic item of a digit that is
left with nothing to grade drops out, and the digit's other items share its weight in proportion. Everything else is
graded as usual: a part whose place is not located is graded for its shape and its holes, the mass, fit and interference
items read every part, piece and hinge where you put it, and the fingertip and coupling items move your digit with all
of your hinges, located or not.

**Palm and alignment.** Parts joined by Fixed joints form one rigid group, and so do the members of an App::Part
container or of a rigid sub-assembly. A group is a palm group when the palm's made parts and circuit boards in it
outweigh, by the volume of the true parts, the digits' made parts in it that carry a servo, an output horn, a bearing or
a dowel pin. The palm is the set of all palm groups, whether joints connect them or not: a palm in several groups loses
only the connection between them. A document has no usable palm when it has no palm group, or when its palm groups hold
none of the pieces and parts that the alignment reads. Your document is aligned with the reference by one rigid motion,
without scaling or mirroring, on the palm alone: on the servos and circuit boards fixed in a palm group, each as placed,
and on the palm's made parts through their fitted poses, each of them only against installed places that the photographs
locate. A servo counts there by the line of its output axis, either way round and wherever along it, so that a servo
mounted the other way round does not turn the hand. After that nothing is resized or moved into place.

**Hinges.** Every Revolute joint between two rigid groups is a hinge on the z axis of its joint frames at the saved
pose; frames that lie more than 0.2 mm or 0.5° apart are reported and read on their mean axis. A Cylindrical joint is
read in the same way, its slide ignored. A Ball joint is read only where it stands for a hinge that turns on a dowel
pin: its centre is a point of that hinge's axis, and the axis is taken parallel to the independent hinge of the same
closed chain. Joints of other types are ignored, and so is a joint inside one group. Several joints between the same two
groups, the palm groups counting as one group, are one hinge on their mean axis when every two of them agree: the frame
origin of each lies within 0.5 mm of the other's axis, and their axes lie within 2° of each other, either way round (a
Ball joint has no axis of its own: its centre must lie within that distance of the other joint's axis, or of its
centre). Joints that do not agree declare no hinge: the hinge between the two groups is missing (below) and is reported,
and none of the joints is preferred; the two groups stay joined, in the connection item and in the order of the digit's
links. A group that holds no servo, no palm part and no made part that carries a servo, an output horn, a bearing or a
dowel pin (a dowel pin or a bearing kept as a body of its own, for instance), and that two joints on one axis join to
exactly two other groups, is passed through: the two joints count as one hinge between its neighbours. A hinge that
turns on a dowel pin and has no joint is read from the pin: when each of two groups either holds the pin by Fixed joints
or has a hole of a made part, or a bearing, on the pin's axis where the pin lies, within the same tolerance, the pin is
a hinge between them. Such a hinge moves like any other, but it is no first hinge of a digit, stands for no servo-driven
hinge and connects nothing in the connection item.

**Digits.** With the palm groups set aside, every set of groups that hinges connect and that a Revolute or Cylindrical
joint joins to a palm group is a candidate digit; that joint is its first hinge. Where more than one such joint joins a
candidate to the palm, the first hinge is the one into the group that holds a made part, of those that carry a servo, an
output horn, a bearing or a dowel pin, of the link nearest the palm; among equals, the joint that comes first in the
document. The candidates are matched one to one to the hand's digits by where the first hinge lies: the distance from
the centre of a servo-driven hinge of the true digit to the axis of your first hinge, plus 0.5 mm per degree of angle
between that axis and the true hinge's, admissible up to 15 mm and 30°. A candidate that fits no digit is graded as
loose parts. Groups that no hinge joins to the palm belong to no digit: their parts keep their shape, hole and placement
credit and lose the connection.

**Links.** A link is a set of parts that move as one. Each link of the true digit is matched to the group, among the
digit's groups and the palm, that holds more of the link's made parts that carry a servo, an output horn, a bearing or a
dowel pin than any other group does, by count; other made parts and given parts never decide a match. Where several
groups hold equally many of a link's such parts and none holds more, the link takes the one of them that no other link
is matched to (the palm counts as taken; tied links are settled in the order of the digit's links); the larger true
volume of those parts decides only among several such groups, or when there is none. Consecutive links matched to one
group form a run. A link none of whose such parts is in the digit or the palm is absent. A part fixed into another link
than in the hand is graded where it is: its placement, the mass and the collisions follow. A hinge of the hand is
present when a hinge of yours joins the groups of the links it connects; it is missing when both links are in one run,
one of them is absent, no hinge joins their groups, or the joints between their groups disagree. A missing hinge is held
rigid at the saved pose and gets no credit for its own items. A digit is invalid only when its structure contradicts the
order of the links from the palm outwards: one group is matched to links that are not neighbours while a link between
them sits in another group, or your hinges reach a later link's group before an earlier one's, unless that later group
holds such a part of the earlier link (a part fixed into the wrong link, which costs its own items only). An invalid
digit, and a digit of the hand that no candidate is matched to, lose their kinematics, mass, fit and interference items;
their parts are still graded for shape, holes and placement.

**Servo-driven hinges.** The driver of a hinge is the servo in one of the rigid groups the hinge joins whose output axis
lies within 5° and 2 mm of the joint's axis; of several, the one whose output centre is nearest the joint frame. The
hinge is then read on that servo's output axis; a joint axis more than 1° or 0.5 mm off it is reported. The servo's own
frame is on its data page: +z is the output axis, out of the output horn, and a positive command turns the horn about +z
relative to the case. The positive sense of the hinge is that rotation when the servo's case is in the palm-side group,
and the opposite when the case is in the fingertip-side group; a case on the other side than in the hand is reported,
and the sense follows the side. A servo-driven hinge is never read from geometry: it needs its Revolute or Cylindrical
joint. Without a driver the hinge is undriven and is treated like a missing one. A servo that no Fixed joint holds is in
no group; where a hinge is declared on such a servo itself, the digit ends there. Every zero is the saved pose.

**Limits.** The declared interval of an independent hinge is read from the Min and Max angle of its joint, taken as
angles of the joint's second frame against its first: the rotation from the x axis of the first frame to the x axis of
the second frame, about the z axis of the first frame. The angle the joint has at the saved pose, measured in the same
way, is subtracted, and the interval is turned into the hinge's positive sense, whichever of its parts the joint
references first and whichever way its frames point. An interval of a full turn or more reads as the full turn. A hinge
without both limits enabled scores 0 on its range item.

**Closed chains.** Where hinges close a chain of parts in the hand, your chain is closed when the part that closes it is
a rigid group of its own, every hinge of the chain is present and the hinge axes are parallel within 2°; a Ball joint on
a dowel pin, or a rod between two Ball joints, closes a chain as well. The grader then solves the chain itself, in the
plane normal to the axis of the chain's independent hinge, from the hinge axes at the saved pose: as that hinge turns,
in steps of at most 2°, the chain keeps the configuration it has at the saved pose. Else the chain is open, and the
hinges that only the chain moves are held rigid at the saved pose. The part that closes a chain never makes a digit
invalid.

**Fingertip point.** The fingertip point of a digit is the sensing-surface centre of its tactile module (a fingertip
tactile module on a finger, the thumb tactile module on the thumb; the point is on the module's data page), carried by
where the module is placed and by the group it is fixed in. Of several such tactile modules in a digit, the one read is
the one fixed in the group that is matched to the link farthest from the palm; among equals, the one that comes first in
the document. A digit without one has no fingertip point: its fingertip item, the length to its fingertip point and its
digit floor are 0, and the digit stays valid.

The design score has seven criteria:

| Criterion | Weight | What it looks at |
|---|---:|---|
| Shape | 0.25 | every made part against the true part of its shape |
| Kinematics | 0.30 | each digit's hinge axes, key lengths, fingertip positions over poses, coupling and ranges |
| Holes | 0.20 | the round holes of the made parts |
| Assembly | 0.10 | where the made and the given parts sit, and what the joints connect |
| Mass | 0.05 | mass and centre of mass of every rigid link |
| Fit | 0.05 | gaps and overlaps of the given and the made parts |
| Interference | 0.05 | collisions of the posed hand |

**Shape.** Each made part against the true part of its shape, in the part's own fit. Surface distance, both ways, 90th
percentile: full credit up to 0.13 mm, none from 1.3 mm (for P01-P05, P16: 0.2 mm and 2 mm); 60 % of the part's score.
Only true surface that the photographs show enters the distance. Volume overlap (intersection over union): none up to
0.3, full from 0.95; 40 %. A part whose surface cannot be meshed as a closed volume of its own size is replaced by its
convex hull in every volume measure (overlap, penetration, collision). A missing part scores 0. Mean over the 52 parts,
each weighted by the square root of its true surface area.

**Kinematics.** Per digit, in the aligned frame; the digits count equally. A digit's items:

- Hinge axes (25 %): for every hinge of the true digit that the photographs locate, the distance from its centre to the
  axis of your hinge (full up to 0.3 mm, none from 3 mm) × the angle between the axes, either way round (full up to
  0.5°, none from 5°), at the saved pose.
- Key lengths (15 %), measured inside your own mechanism at the saved pose and compared with the hand's: the distance
  between consecutive hinge axes along the digit and between the hinge axes of a closed chain (the common normal; for
  axes within 5° of parallel, the mean distance of each from the other), the angle between consecutive hinge axes along
  the digit, and the distance of the fingertip point from the last hinge axis. Distances: full up to 0.3 mm of error,
  none from 3 mm; angles: 0.5° and 5°. A length that needs a missing hinge or a missing fingertip point gets no credit.
  A length or angle that ends at a hinge the photographs do not locate is not graded.
- Fingertip over poses (35 %; 45 % in a digit of the hand without a closed chain): 64 poses of the digit's independent
  hinges inside their true ranges, drawn afresh for every grading and kept where the true digit collides neither with
  itself nor with the palm. Your digit is posed at the same hinge values with its own axes, senses and closed chain, and
  the error is the distance between your fingertip point and the true one: 50 % median (full up to 1 mm, none from 6
  mm), 50 % 90th percentile (full up to 1.5 mm, none from 10 mm).
- Coupling (10 %, in a digit of the hand with a closed chain): at 32 angles of the chain's independent hinge across its
  true range, the mean difference between the angle that your chain gives its dependent hinge on the path from the palm
  to the fingertip and the true angle (full up to 1.5°, none from 12°). An angle at which your chain cannot close counts
  as the far end of that band, and an open chain scores 0.
- Range (15 %): for every independent hinge of the true digit that the photographs locate, the overlap of your declared
  interval with the true interval, divided by their union (an end within 0.01° of the true end counts as equal); 0 for a
  hinge that is missing, undriven or without both limits enabled.

**Holes.** The round holes of the made parts. A graded step is one concave cylindrical wall of the true part that covers
at least 300° about its axis, with a radius up to 12 mm and a depth of at least 0.4 mm; in a stepped hole each such step
is graded. The steps are matched one to one to the concave cylindrical walls of your part, in the part's own fit. A wall
of yours is eligible when it covers at least 300° and is at least as deep as the larger of 0.4 mm and 50 % of the true
step's depth; coaxial faces of one radius count as one wall, and a face that is no exact cylinder counts when it fits
one within 6 % of the radius (at least 0.02 mm). Such a fitted face covers the whole turn when it closes on itself
around its axis, else the arc that its surface spans, and fitted faces join into one wall when their circles agree
within that same tolerance. Walls of yours that cover at least 300° and whose axes lie within 0.9 mm and 10° of each
other, directly or through other such walls, are one hole of yours. Such a hole stands at a true hole when one of its
walls is eligible for a step of the true hole and lies within that distance and angle of it. When it then has more walls
that are eligible for the true hole than the true hole has steps, only as many of them as the true hole has steps are
read for it, chosen by their depth and never by how well they would score: the deepest first (depths within 0.01 mm of
the deepest count as equal to it), then the larger radius, then the order of the walls in your part. The walls left over
are not read for that hole, and each of them that scores for no other hole is an extra (below). A matched step scores
its axis offset at the step's mid-depth (full up to 0.15 mm, none from 0.9 mm) × radius error (full up to 0.08 mm, none
from 0.4 mm) × axis angle (full up to 2°, none from 10°). A hole scores the mean of its steps, a part the weighted mean
of its holes that the photographs show: weight 2 for a hole in which a bearing or a servo's rear bearing sits, through
which a dowel pin passes, or that lies against a servo's output horn on the output axis or on the horn's bolt circle;
weight 1 for every other hole. An extra hole is not free: a wall of yours is an extra when it is a cylinder that covers
at least 300°, is at least 0.4 mm deep and has a radius from 0.3 mm to 6.05 mm, scores for no true step, and cannot be
paired, one wall to one step, with a true step of the part that no wall of yours scores for and that it meets within 0.9
mm (its axis at the step's mid-depth) and 10°, whatever its radius. Each extra adds to the divisor of the part's
weighted mean the largest weight that a single step of the part's graded holes carries (a hole's weight divided by the
number of its steps). A wall paired in this way with a step of a true hole that the photographs do not show earns
nothing and costs nothing. So a hole where the part has none, a second copy of a hole at another place, and a hole
farther than that from where it belongs each lower the part's hole score, and a hole drilled at each of two possible
places scores, on average, no more than a hole at the likelier one. Mean over the parts with such holes, weighted like
shape. Walls that do not close (open slots, half-round recesses) are no holes; they belong to the shape.

**Assembly.** 50 % placement of the made parts whose installed places the photographs locate: each part's fitted pose
against its true pose in the aligned frame, position of the part's centroid (full up to 0.5 mm, none from 5 mm) × (0.7 +
0.3 × orientation, full up to 2°, none from 15°); a pose that a symmetry of the part maps onto the true one counts as
exact; weighted like shape. 30 % installation of the given pieces whose installed places the photographs locate:
position of the piece's origin (full up to 0.5 mm, none from 5 mm) × (0.7 + 0.3 × orientation, full up to 1°, none from
10°), a symmetric pose counting as exact, with the grading weights of the table of given parts. 20 % connection: the
weighted share (made parts weighted like shape, given pieces by their grading weights, each of the two kinds counting
half) of the largest set of rigid groups that joints of the types Fixed, Revolute, Cylindrical and Ball connect and that
contains a palm group.

**Mass.** Per rigid link of the true hand (the palm with weight 4, every link of a digit with weight 1): 50 % mass error
(full up to 2 %, none from 15 %) + 50 % centre of mass at the saved pose in the aligned frame (full up to 0.5 mm, none
from 5 mm), measured on the rigid group of yours that is matched to the link: all palm groups together for the palm, a
run against the sum of its links; a link that is absent scores 0. Mass = solid volume × the material's density for a
made part, the data-page mass for a given piece, extra solids as stated above. Dowel pins and bearings are left out on
both sides, so it does not matter which of their parts holds them.

**Fit.** 60 %: every given piece, with its grading weight: the median gap over the patches where the piece touches made
parts in the hand, read only on true surface that the photographs show (full up to 0.1 mm, none from 1 mm) × the share
of its volume that lies inside made parts and extra solids (full up to 0.05 %, none from 5 %); a piece that touches no
made part in the hand, or whose seat the photographs do not show at all, is scored on the second term alone. 40 %: every
made part: the share of its volume inside other made parts, given pieces and extra solids, on the same band. What the
hand itself has of overlap between the same two items is exempt. A missing part or piece scores 0.

**Interference.** Your hand, posed by its own hinges at the poses of the fingertip item; the palm groups count as one
group. A pose passes when no two groups overlap by more than at the saved pose plus 1 % of the smaller group's volume
(groups that a hinge joins) or 0.1 % (any other pair). 75 %: the share of passing poses among the groups of the digit
and the palm; 25 %: between the groups of different digits, at the poses where the whole true hand is free of collision.

**Floors.** Kinematics, holes, assembly, mass, fit and interference are multiplied by a shape floor that rises linearly
from 0 at a mean shape score of 0.04 to 1 at 0.6: correct joints and given parts around crude made parts collect little
beyond the shape credit. Inside mass, fit and interference, the items of a digit are also multiplied by that digit's
floor, which falls linearly from 1 at a median fingertip error of 3 mm to 0 at 12 mm (0 for an invalid digit and for a
digit without a fingertip point); the palm's items take the mean of the digits' floors. design_score = 0.25 shape +
shape floor × (0.30 kinematics + 0.20 holes + 0.10 assembly + 0.05 mass + 0.05 fit + 0.05 interference); the reward is
the design score. The unit itself, modelled exactly, scores 1.

## Practical notes

- Session time: the session is stopped 6 hours after its start, and the document that is saved in
  `/home/user/submission` at that moment is graded as it stands: work that is not saved by then is lost. Note the time
  when you begin, save a complete document early and keep saving it, and plan to finish at least 15 minutes before the
  limit.
- A single model response that reaches 64k output tokens is lost: keep every response well below that, and do not wait
  until the end to create the deliverable.
- `execute_code` returns after 90 s while the call keeps running: when a call reports a timeout, look at the document
  before you repeat it.
- Creating a joint lets FreeCAD's solver move the second part until the two joint frames coincide. Check the placements
  after adding joints and before the last save, above all where joints close a chain of parts. FreeCAD does not stop a
  hinge at its angle limits: they are read as declared.
- The machine has 8 CPUs and 16 GB of memory; many full-size photos held in memory at once can exhaust it.
