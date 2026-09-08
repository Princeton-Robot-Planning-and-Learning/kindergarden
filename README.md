# Dynamic3D draft PR review media

GIFs for the September 7, 2026 draft reviews. This branch contains review media only.

| File | Evidence |
|---|---|
| dynamo.gif | Same three-chair seed1 reset before/after camera patch4f67aad. |
| balance.gif | Final placement segment from saved legal controller actions on1907f6f, then40 successful settling actions. |
| sort.gif | Normal-reset o4 seed0 saved-action replay on303c8ce; released placement and settling. |
| rearrange.gif | Final placement and60 settling actions on739d4b8; bowl stays on counter. Accelerated2.5x. |
| drawer.gif | Before/after API restoration of a reachable closed snapshot after180 legal opening actions,8f2fcd9 versusedf8f3f. Snapshot restore demo, not a full task solution. |
| tossing-farther.gif | Both throws from normal-reset o2 seed0 on469372f; original barrier and movable bin.0.5x playback. Extra close-up camera is visualization only. |

These clips do not establish complete seed/count coverage. The associated draft PR bodies state validation limits. In particular, moving the tossing bin does not make tossing mandatory, and no complete ScoopPour controller success is claimed.

## September 8 follow-up

- `tossing-reach-drop-before-after.gif`: original o2 seed0 reach/drop controls on56369da versus farther box469372f. Old layout succeeds366+60settling; same426controls fail with farther box. Fixed-action comparison, not all-policy impossibility evidence.
- `tossing-farther-no-toss-bypass.gif`: farther-bin o1 seed0 still solved by driving around barrier and placing. Success357+60settling, full cube contained; wide navigation view then closeup. This shows why distance alone does not require tossing.

Both new clips show4x playback and exact step/goal labels. Public code review descriptions explain controller scope and limitations.

- `sort-before-no-bin-success.gif`: normal-reset o4 seed0 original controller on original fixed table goals; first success193, final299 after100releasedsteps, all4cubes ontableoutsidebins.8x playback.
- `sort-after-rejects-table-placement.gif`: exact299legalactions on4e7a00f fixed bin-relativegoals; finalgoalfalse.5x playback. These are action-only episodes, not injected states. Same action sequence is verified; bitwise final simulator state comparison is not claimed.

- `tossing-reach-drop-before-after-background.gif`: replaces the plain-scene comparison in#191 at the user's request. Full MimicLabs scene backgrounds (`scene_bg=True`), camera inside the room; exact final goal/containment results match the original plain-scene renders.4x playback.

- `rearrange-five-single-object.gif` and `rearrange-five-two-object.gif`: alltenaffected#188instructionvariants, seed0, normal-reset action-only solutions. Eachretains60/60successafterrelease; samecontrols underoriginalgoals0/60, finalobservedphysicalstatesexactlyequal. Fiveinstructionexcerpts perclip,10x playback, fullscene backgrounds; finalplacement excerpts,notfullmovies. No bowlheld/lifted,no finalrobot-objectcontacts;maxfinalbowldisplacement0.29mm.

- `scoop-pour-controller-background.gif`: full-scene o10 seed0 source-tray pour, first correctedgoal780, source trayreleased1164, armwithdrawn1227,100settlingsteps through1327alltrue. All10fullcubeshapes inside receiver; literal former drawerpredicate evaluatedreadonly remainsfalse on samepatchedtrajectory (not separateold-checkoutreplay). Source traydirectlypoured,scooptoolunused. Oneconstructive seed, notsuccessrate.

- `sort-intended-background.gif`: exact459legalactions from normal-reset o4seed0 successfulcontroller, nowrendered withscene_bg=True. Includes100releasedsettlingsteps; finalgoaltrue and priortracecheckedfullcubecorners. Replacesoriginalplainbackgroundsort.gif in#185. Globalpaletteencoding reducesdownloadsize withoutchangingtrajectory.
