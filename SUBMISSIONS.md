# Submission workflow

Members no longer retype scores. Every call to `challenge.train()` records one row in the
organizer's Google Sheet automatically: the result, the code that produced it, and the Google
account email Colab is signed in with. A Google Form still collects the few things the
notebook cannot send.

## How it works

1. `Challenge()` checks the GPU, reads the member's Google account email through Colab's
   OAuth, and asks once for a display name.
2. Each `challenge.train()` run builds a row and POSTs it to an Apps Script web app.
3. The script appends the row to a tab named after the workshop.
4. A public-safe dashboard polls the sheet every 2.5 seconds and smoothly adds the run to
   the competition history chart and leaderboard.
5. If the POST fails, the run is written to `outputs/submissions/*.json` inside the Colab
   session and the training cell says so.

All of this lives in `mandelbrot_challenge/submission.py`, which members never open.

## One-time organizer setup

### 1. Create the sheet

Make a new Google Sheet that you own. Copy its ID from the URL:
`https://docs.google.com/spreadsheets/d/SHEET_ID_IS_HERE/edit`.

Do not share it with members. They never read from it.

### 2. Add the Apps Script

In the sheet choose **Extensions → Apps Script**. The repository contains the two files to
copy into that project:

1. Replace `Code.gs` with [`apps_script/Code.gs`](apps_script/Code.gs).
2. Click **+ → HTML**, name the file `Dashboard`, and replace it with
   [`apps_script/Dashboard.html`](apps_script/Dashboard.html).
3. In `Code.gs`, fill in `SHEET_ID` and `SHARED_TOKEN`.
4. Keep the active workshop in `PUBLIC_WORKSHOP_IDS`. Only those tabs can appear on the
   public dashboard.

The dashboard receives only member name and performance numbers. Email addresses, submitted
code, optimizer/loss descriptions, and the shared token never leave the private sheet.

### 3. Deploy it

**Deploy → New deployment → Web app**, then:

- Execute as: **Me**
- Who has access: **Anyone**

Authorize when prompted and copy the `/exec` URL.

After changing an existing deployment, choose **Deploy → Manage deployments → Edit**, select
**New version**, and deploy. Saving the files alone does not update the public `/exec` URL.

"Anyone" is required because members' Colab runtimes post without a Google identity attached
to the request. The script is append-only and token-guarded, so the worst an outsider can do
is add junk rows.

### 4. Point the package at it

Edit the three constants in `mandelbrot_challenge/submission.py` and push to `main`, so the
notebook's install step picks them up:

```python
DEFAULT_ENDPOINT = "https://script.google.com/macros/s/.../exec"
DEFAULT_WORKSHOP_ID = "2026-fall-mandelbrot"
DEFAULT_TOKEN = "the same random string you put in the script"
```

Bump `DEFAULT_WORKSHOP_ID` before each new workshop. Each value gets its own tab, so one sheet
serves many workshops.

For testing, `MANDELBROT_SUBMIT_URL`, `MANDELBROT_WORKSHOP_ID`, and `MANDELBROT_SUBMIT_TOKEN`
override the constants without editing code.

### Open the live dashboard

Open the same `/exec` URL in a browser. The current workshop is selected automatically, or
you can link to it explicitly:

```text
https://script.google.com/macros/s/DEPLOYMENT_ID/exec?workshop=2026-fall-mandelbrot-beta
```

Successful `print_result(result)` output includes this link. Open the deployed `/exec` URL,
not the raw `Dashboard.html` file: the raw file has no permission to read the Google Sheet.
The page checks for a new row every 2.5 seconds; no refresh button is needed. Each dot is a
run, orange dots are new public records, and the orange staircase is the best score seen so
far. The leaderboard ranks the 10 lowest public validation MAE results. Lower MAE is better.

### What the token does and does not do

The POST happens inside the member's own Colab runtime, so the endpoint and token are always
recoverable by a member who goes looking. They route requests and stop drive-by traffic; they
are not a security boundary. Integrity comes from the server being append-only, from the
verified email on each row, and from rerunning finalists yourself.

## What each run records

Identity: `workshop_id`, `member_name`, `member_email`, `email_source`, `submitted_at`.

Result: `validation_mae`, `parameter_count`, `training_seconds`, `steps`,
`samples_per_second`, `gpu_name`.

Code: `model_source`, `model_repr`, `optimizer_repr`, `loss_repr`, `scheduler_repr`,
`batch_size`.

Environment: `package_version`, `torch_version`, `run_stamp`.

`email_source` is `colab_oauth` when Google supplied the address and `manual` when the member
typed it because OAuth was declined or unavailable. Treat `manual` rows as unverified.

Model weights are never sent. The notebook discloses this list to members before they train.

## The Google Form

Keep the form for what the notebook cannot send. It is now short:

1. View-only Colab notebook link
2. Cleared-output `.ipynb` upload
3. Required checkbox: "I changed only the two member cells, did not calculate the Mandelbrot
   formula in my model, did not train outside the official timer, and this submission contains
   the code I ran."

Settings: collect email addresses, limit to one response, allow editing until the deadline,
restrict uploads to `.ipynb`, close at the deadline. Match the form email against
`member_email` in the sheet.

## Member steps

1. Run the notebook from the top. Approve the Google sign-in prompt and type a display name.
2. Edit the two `MEMBER WORKSPACE` cells and rerun the training cell as often as you like.
   Every run is recorded.
3. Share the notebook as *Anyone with the link → Viewer*.
4. Save a backup copy, clear its outputs, download the `.ipynb`, and submit it with the link
   on the form.

## Organizer scoring

1. Open the workshop's tab. Sort by `member_email`, then `validation_mae`, and take each
   member's best row.
2. Treat every recorded score as provisional.
3. Read `model_source` and the optimizer, loss, and scheduler columns for rule violations.
4. Rerun qualifying submissions from a fresh runtime on the same NVIDIA T4.
5. Run the private evaluator on the resulting in-memory `model`.
6. Sort by hidden MAE rounded to five decimals, then parameter count, then inference time.
7. Keep the private evaluator and organizer benchmark inaccessible until the competition and
   any appeals are complete.

Add organizer-only columns to the right of the recorded ones: `Eligible`, `Hidden MAE`,
`Verified parameters`, `Inference ms`, and `Review note`. The script only writes columns it
knows about, so yours are left alone. Publish only member name, rounded hidden MAE,
parameters, and rank unless members consent to sharing more.

## When something goes wrong

**A member's rows are missing.** The endpoint was unreachable or the token is stale. Their runs
are in `outputs/submissions/*.json` in the Colab session. Ask them to open that folder in the
Colab file browser and send you the files before the runtime disconnects.

**`email_source` is `manual`.** The member declined the Google prompt or Colab auth failed.
The email is self-reported; confirm it against the form response.

**Duplicate members.** Dedupe on `member_email`, not `member_name`. Members may type their
name differently between sessions; the email is stable.

**Rows land in an `unsorted` tab.** A run posted without a `workshop_id`, which means an old
package version. Confirm the member reinstalled from `main`.
