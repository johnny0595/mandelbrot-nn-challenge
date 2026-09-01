const SHEET_ID = 'PASTE_YOUR_SHEET_ID_HERE';
const SHARED_TOKEN = 'PASTE_A_LONG_RANDOM_STRING_HERE';

// Only these workshop tabs may be read by the public dashboard. Add the next
// workshop here when DEFAULT_WORKSHOP_ID changes in submission.py.
const PUBLIC_WORKSHOP_IDS = Object.freeze([
  '2026-fall-mandelbrot-beta',
]);

function doPost(e) {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(30000)) {
    return jsonReply({ ok: false, error: 'busy' });
  }

  try {
    const body = JSON.parse(e.postData.contents);
    if (body.token !== SHARED_TOKEN) {
      return jsonReply({ ok: false, error: 'bad token' });
    }

    const payload = body.payload;
    if (!payload || !payload.member_email) {
      return jsonReply({ ok: false, error: 'missing member_email' });
    }

    const sheet = sheetFor(payload.workshop_id || 'unsorted');
    const runNumber = appendByHeader(sheet, payload);
    SpreadsheetApp.flush();
    return jsonReply({ ok: true, run_number: runNumber });
  } catch (error) {
    return jsonReply({ ok: false, error: String(error) });
  } finally {
    lock.releaseLock();
  }
}

// Opening the /exec URL shows the live progress dashboard. The query parameter
// selects a public workshop, for example: ?workshop=2026-fall-mandelbrot-beta
function doGet(e) {
  const workshopId = publicWorkshopId(e && e.parameter && e.parameter.workshop);
  const template = HtmlService.createTemplateFromFile('Dashboard');
  template.workshopId = workshopId;
  return template.evaluate()
    .setTitle('Mandelbrot challenge progress')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function jsonReply(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function publicWorkshopId(requested) {
  const candidate = String(requested || PUBLIC_WORKSHOP_IDS[0]);
  return PUBLIC_WORKSHOP_IDS.indexOf(candidate) === -1
    ? PUBLIC_WORKSHOP_IDS[0]
    : candidate;
}

function sheetFor(name) {
  const book = SpreadsheetApp.openById(SHEET_ID);
  return book.getSheetByName(name) || book.insertSheet(name);
}

// Place each value under its named column. Organizer-only columns to the right
// remain untouched when later submissions arrive.
function appendByHeader(sheet, payload) {
  const width = Math.max(sheet.getLastColumn(), 1);
  let headers = sheet.getLastRow() === 0
    ? []
    : sheet.getRange(1, 1, 1, width).getValues()[0].filter(String);
  const missing = Object.keys(payload).filter(key => headers.indexOf(key) === -1);

  if (missing.length) {
    headers = headers.concat(missing);
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
    sheet.setFrozenRows(1);
  }

  sheet.appendRow(headers.map(key => (payload[key] === undefined ? '' : payload[key])));
  return sheet.getLastRow() - 1;
}

// Called from Dashboard.html with google.script.run. This is deliberately an
// allowlist: emails, source code, model descriptions, and tokens never reach the
// browser even though they remain available to the organizer in the private sheet.
function getDashboardData(requestedWorkshopId) {
  const workshopId = publicWorkshopId(requestedWorkshopId);
  const book = SpreadsheetApp.openById(SHEET_ID);
  const sheet = book.getSheetByName(workshopId);
  if (!sheet || sheet.getLastRow() < 2) {
    return { workshopId: workshopId, runs: [], generatedAt: new Date().toISOString() };
  }

  const values = sheet.getDataRange().getValues();
  const headers = values[0].map(String);
  const column = {};
  headers.forEach((header, index) => { column[header] = index; });

  const read = (row, name) => column[name] === undefined ? '' : row[column[name]];
  let runningBest = Infinity;
  const runs = [];

  values.slice(1).forEach((row, rowIndex) => {
    const score = finiteNumber(read(row, 'validation_mae'));
    if (score === null) return;

    const isRecord = score < runningBest;
    runningBest = Math.min(runningBest, score);
    const submittedAt = read(row, 'submitted_at');
    runs.push({
      number: rowIndex + 1,
      submittedAt: submittedAt instanceof Date ? submittedAt.toISOString() : String(submittedAt || ''),
      memberName: String(read(row, 'member_name') || 'Anonymous'),
      validationMae: score,
      parameterCount: finiteNumber(read(row, 'parameter_count')),
      trainingSeconds: finiteNumber(read(row, 'training_seconds')),
      steps: finiteNumber(read(row, 'steps')),
      samplesPerSecond: finiteNumber(read(row, 'samples_per_second')),
      batchSize: finiteNumber(read(row, 'batch_size')),
      isRecord: isRecord,
      runningBest: runningBest,
    });
  });

  return {
    workshopId: workshopId,
    runs: runs,
    generatedAt: new Date().toISOString(),
  };
}

function finiteNumber(value) {
  if (value === '' || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
