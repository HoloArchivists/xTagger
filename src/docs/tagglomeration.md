# Tagglomeration: teaching xtag new tags

Tagglomeration teaches wd tags that no model has, such as a character newer than both models, from a few images you
label yourself. You put a text file beside an image, list its tags, and run with `-g`. xtag then learns those tags and
applies them to every other image it tags, in the same run and in later runs.

This guide walks through a complete example: teaching `gangle_(the_amazing_digital_circus)` from 4 labelled images in
a folder of 87, of which 11 show Gangle.

## 1. Write sidecars

A sidecar is a `.txt` file with the same name as the image, in the same folder:

```
gel_rand_hard\
  haha.png
  haha.txt        <- tags for haha.png
  sowwy.png
  sowwy.txt
```

Inside, list tags separated by commas or line breaks:

```
character:gangle_(the_amazing_digital_circus), copyright:the_amazing_digital_circus
```

- **Names are normalised.** Spaces become underscores and letters are lowercased. Backslashes are dropped, so
  `gangle_\(the_amazing_digital_circus\)` and `Gangle (The Amazing Digital Circus)` both mean the same tag.
- **A prefix sets the category** of a new tag: `character:`, `copyright:`, `artist:`, `meta:` or `general:`. The
  category decides the threshold (`-tc` for characters, `-tg` for everything else). Without a prefix a new tag is
  general. A prefix on any later mention changes the category.
- **What happens to each name depends on who knows it:**
  - **wd already has the name:** wd's tag is pushed toward "yes" on this image.
  - **Only PixAI has it** (run with `-px`): the tag is taken from PixAI (stolen) and seeded from this image. Add `-s`
    so PixAI keeps refining it on later images.
  - **Neither model has it:** it becomes a new user tag, learned from your sidecars. Gangle is this case.
- **A sidecar is taken as complete for user tags.** An image whose sidecar does not list `gangle_(...)` is treated
  as "not Gangle". Don't write a sidecar on a Gangle image unless it lists Gangle.

## 2. Choose the examples

- **Label 3 to 5 images of each look.** Gangle has a happy mask and a sad mask, so label both. The 4 labelled
  images in the example are two happy and two sad.
- **Leave the rest unlabelled.** You don't need to label every image of the subject; finding the others is the
  point. Unlabelled Gangle images are not trained against.
- **Label lookalikes as negatives,** but only once you have seen one fire. A lookalike is an image without the
  subject that still triggers the tag. Give it a sidecar that leaves the tag out; anything else it lists is fine:

  ```
  gelbooru_15173767_8651ad313790e04b65a5e4be006b2441.txt:
  yellow_shirt, 1girl
  ```

  In the example that image is a shirt printed with a cartoon face, and this one line stopped it firing.

## 3. Run it twice

### Learning run

```powershell
.\build\xtag.exe .\src\models\eva02.bin .\src\models\selected_tags.csv "C:\Users\101120541\Downloads\gel_rand_hard\" -mc 16 -t 448 -rT auto -px .\src\models\pixai.bin -g -gf learned.bin -s -sk cp -sT > run1.txt
```

- `-g` turns tagglomeration on.
- `-gf learned.bin` keeps what it learns across runs (and turns `-g` on by itself).
- `-s -sk cp` adds PixAI tag stealing. It is optional; tagglomeration works without `-px` and `-s`.
- The `-sT` board at the end should show `sidecar tags      1 held, 4 images`: one user tag held, four sidecar
  images read. If it says 0 images, the `.txt` names don't match the image names.

Learning needs a mean of 8 images, so xtag puts the sidecar images right after the first 8 others. Those first 8
are tagged before anything is learned. Expect a few misses among them on the first run.

### Tagging run

```powershell
.\build\xtag.exe .\src\models\eva02.bin .\src\models\selected_tags.csv "C:\Users\101120541\Downloads\gel_rand_hard\" -nc -rT wd -gf learned.bin -sT > run2.txt
```

- This run loads `learned.bin` and tags every image with the finished tag.
- `-gf` keeps learning on, so later runs (and new sidecars) keep improving the file.
- To tag without changing the file, copy it first and point `-gf` at the copy.

### Results in the example (measured)

| Sidecars | Learning run | Tagging run |
|---|---|---|
| 4 Gangle images | 8 of 11 Gangle, 1-2 false | 11 of 11, 1-3 false |
| + 1 negative on the cartoon-face shirt | 8 of 11, 0 false | 11 of 11, 0-1 false |

The learning run's misses are three Gangle images among the first 8, tagged before learning started. The false
positives are lookalikes, and a negative sidecar removes each one.

## 4. Check the result

PowerShell's `>` writes UTF-16. `Get-Content` reads it correctly; other tools may need converting first. To list
the images that got a tag:

```powershell
$tag = 'gangle_(the_amazing_digital_circus)'
Get-Content run2.txt | ForEach-Object {
    $c = $_ -split "`t"
    if ($c.Count -ge 4 -and ($c[3] -split ',' | Where-Object { $_.StartsWith("$($tag):") })) { Split-Path $c[0] -Leaf }
}
```

To write UTF-8 instead, end the command with `| Out-File -Encoding utf8 run2.txt` in place of `> run2.txt`.

## 5. Tune

| Problem | Fix |
|---|---|
| A lookalike gets the tag | Give it a sidecar without the tag (section 2). This is the main fix. |
| Images of the subject are missed | Label one or two of them (their look is probably not covered), or lower `-gk` |
| Too many false positives everywhere | Raise `-gk` (default 3; 3.5 or 4 is stricter) |
| The tag fires on nothing | Check `-sT` shows the sidecar images; at least 8 images must be tagged before learning starts |
| Score near the threshold | `-gk` sets the cutoff and `-tc` / `-tg` only shift where it is printed; prefer `-gk` |

`-gk` is measured in standard deviations of the background, meaning every image not labelled with the tag. At 3, the
tag fires when an image is closer to your examples than all but the most similar background images. On the example
folder, 3 found all 7 unlabelled Gangle images and none of the other 76.

## 6. Things to know

- **One file holds everything.** Stolen tags and user tags live in the same file: `-sf` and `-gf` must name the same
  file, and giving two different files is an error. Without `-gf` / `-sf`, nothing is kept after the run.
- **Start fresh with a new file name.** Files from before this version load, and their user tags are rebuilt from
  the stored examples. If results look off, start a new file.
- **Every image costs one extra wd pass.** While user tags exist, every image gets wd's whole-image GPU pass,
  whatever the route, because user tags are scored from it. wd is the fast model, but this still costs.
- **The NPU is not used while user tags exist.** Only the GPU pass produces the embedding the user tags need.
- **Crops don't count.** User tags are scored on the whole image only.
- **Sidecar images run every route.** With `-px` they run both models and all crops, so they are the slowest images
  in the run.
- **Rerunning a sidecar image adds nothing new.** It reuses its stored example, and its labels are merged.
- **Limits:** 256 user tags, 255 stored sidecar images, and new rows (stolen and user tags together) up to `-sn`.

## 7. How it works

Each user tag compares an image's wd embedding with the average of your labelled images (more precisely, the cosine
after subtracting the mean image). That score is then judged against the background:

- **Cutoff:** the background's median plus `-gk` standard deviations, measured robustly so unlabelled images of the
  subject don't move it.
- **Several looks:** a tag can hold up to 4 groups of examples, one per look, and fires if any group matches.
- **Grouping:** a small genetic search (NEAT-style) decides which examples group together and which negatives to
  subtract. It scores each grouping by how well every example is recognised with itself left out, so it can't
  succeed by memorising.

The source is `src/core/tagg.h`, and its header comment has the details.
