# run_eval_chain.ps1 -- chaine d'evaluation J4-bis, supervisee et idempotente.
#
# Concu pour etre lance DETACHE du shell de l'agent (AGENTS.md 11.1) :
# tache planifiee Windows -> ici "Jainius-EvalJ4bis".
#
# Garanties :
#  - heartbeat horodate toutes les 60 s (heartbeat.log) : vivant, PID python,
#    memoire/utilisation GPU, dataset courant ; s'arrete avec le superviseur ;
#  - journal principal (chain.log), codes de sortie par dataset
#    (chain_state.json), stdout/stderr par tentative (run_<ds>_<ts>.log/.err) ;
#  - idempotence : un dataset dont le JSON est deja valide est saute ;
#  - faulthandler active cote python (dump memoire sur faute dure) ;
#  - post-mortem : evenements Windows (python/powershell/nvlddmkm) depuis le
#    debut de la chaine -> events_chain_<ts>.log.
#
# Codes de sortie : 0 = chaine complete ; 2 = datasets manquants (la tache
# planifiee relance automatiquement) ; 3 = erreur fatale.
#
# Controle utilisateur (GPU) :
#   arret  : Disable-ScheduledTask -TaskName Jainius-EvalJ4bis
#            puis Stop-ScheduledTask -TaskName Jainius-EvalJ4bis
#            (verifier qu'aucun python du run ne subsiste : Get-Process python)
#   reprise: Enable-ScheduledTask -TaskName Jainius-EvalJ4bis
#            puis Start-ScheduledTask -TaskName Jainius-EvalJ4bis
#            (saute les datasets deja valides)
#   etat   : Get-ScheduledTaskInfo -TaskName Jainius-EvalJ4bis

param(
  [string]$Ckpt = "_artefacts/j4b/phase2/step_000979",
  [string]$OutDir = "_artefacts/j4b",
  [string]$Agg = "_artefacts/j4b/eval_j4b.json",
  [string[]]$Datasets = @("gsm8k","math500","mmlu","mmlu_pro","humaneval","mbpp"),
  [int]$HeartbeatSeconds = 60,
  [switch]$SkipPytest,
  [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDirAbs = (Resolve-Path $OutDir).Path
$chainLog = Join-Path $OutDir "chain.log"
$hbLog = Join-Path $OutDir "heartbeat.log"
$stateFile = Join-Path $OutDir "chain_state.json"
$dsFile = Join-Path $OutDir "current_ds.txt"

function Log([string]$msg) {
  $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $chainLog -Value $line
  Write-Host $line
}

# --- python (resolution explicite, journalisee) ---
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
  $cand = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
  if (Test-Path $cand) { $py = $cand }
}
if (-not $py -or -not (Test-Path $py)) {
  Log "[FATAL] python introuvable"
  exit 3
}

$gpu = ((nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -join " ")
$apps = ((nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) -join "; ")
Log ("[start] ckpt={0} python={1} gpu={2} compute_apps=[{3}] dryrun={4}" -f $Ckpt, $py, $gpu, $apps, $DryRun)

# --- heartbeat (job enfant ; s'arrete si le superviseur disparait) ---
$supervisorPid = $PID
$hbScript = {
  param($HbFile, $StateFile, $RepoDir, $Seconds, $SupPid)
  Set-Location $RepoDir
  while ($true) {
    if (-not (Get-Process -Id $SupPid -ErrorAction SilentlyContinue)) { break }
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $pyProcs = @(Get-Process python -ErrorAction SilentlyContinue)
    $pids = ($pyProcs | ForEach-Object { $_.Id }) -join ","
    if (-not $pids) { $pids = "-" }
    $g = "n/a"
    $gpuLine = nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
    if ($gpuLine) { $g = ($gpuLine -join " ") }
    $ds = "-"
    if (Test-Path $StateFile) { $ds = (Get-Content $StateFile -Raw).Trim() }
    Add-Content -Path $HbFile -Value ("{0} | alive | python={1} | gpu={2} | ds={3}" -f $ts, $pids, $g, $ds)
    Start-Sleep -Seconds $Seconds
  }
}
$hb = Start-Job -ScriptBlock $hbScript -ArgumentList $hbLog, $dsFile, $repo, $HeartbeatSeconds, $supervisorPid

# --- fonctions d'execution ---
function Test-JsonOk([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  try { $null = Get-Content $path -Raw | ConvertFrom-Json; return $true } catch { return $false }
}

# --- arret rapide si la chaine est deja complete (relance AtLogOn) ---
if (-not $DryRun) {
  $allDone = $true
  foreach ($ds in $Datasets) {
    if (-not (Test-JsonOk (Join-Path $OutDir ("eval_j4b_{0}.json" -f $ds)))) { $allDone = $false; break }
  }
  if ($allDone -and (Test-JsonOk $Agg)) {
    Log "[skip] chaine deja complete (eval_j4b.json present) : rien a faire"
    Stop-Job $hb -ErrorAction SilentlyContinue
    Remove-Job $hb -Force -ErrorAction SilentlyContinue
    exit 0
  }
}

function Invoke-Py([string[]]$pyArgs, [string]$tag) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $outLog = Join-Path $OutDir ("run_{0}_{1}.log" -f $tag, $ts)
  $errLog = Join-Path $OutDir ("run_{0}_{1}.err" -f $tag, $ts)
  if ($DryRun) {
    Add-Content -Path $outLog -Value "[dryrun] would run: $py $($pyArgs -join ' ')"
    Start-Sleep -Seconds 2
    return 0
  }
  $env:PYTHONFAULTHANDLER = "1"
  $env:PYTHONUNBUFFERED = "1"
  $p = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $repo `
    -NoNewWindow -PassThru -Wait -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  return $p.ExitCode
}

# --- boucle datasets (idempotente) ---
$state = [ordered]@{}
foreach ($ds in $Datasets) {
  $outJson = Join-Path $OutDir ("eval_j4b_{0}.json" -f $ds)
  if (Test-JsonOk $outJson) {
    Log "[skip] $ds : resultat deja present et valide"
    $state[$ds] = "skip-existing"
    continue
  }
  if (Test-Path $outJson) {
    $bad = "$outJson.corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Move-Item -Force $outJson $bad
    Log "[warn] $ds : JSON illisible, deplace vers $bad"
  }
  Set-Content -Path $dsFile -Value $ds
  $gpuNow = ((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join " ")
  Log "[ds:$ds] start (gpu_mem=$gpuNow MiB)"
  $t0 = Get-Date
  $code = Invoke-Py @("-X","faulthandler","-u","scripts/eval_j4.py","--ckpt",$Ckpt,"--out",$outJson,"--datasets",$ds) $ds
  $dur = [int]((Get-Date) - $t0).TotalMinutes
  Log "[ds:$ds] end exit=$code duree=${dur}min"
  $state[$ds] = "exit=$code"
}
Set-Content -Path $dsFile -Value "post-datasets"

# --- bilan + agregation + tests ---
$missing = @()
foreach ($ds in $Datasets) {
  $outJson = Join-Path $OutDir ("eval_j4b_{0}.json" -f $ds)
  if (-not (Test-JsonOk $outJson)) { $missing += $ds }
}
if (-not $DryRun -and $missing.Count -eq 0) {
  $ins = @()
  foreach ($ds in $Datasets) { $ins += (Join-Path $OutDir ("eval_j4b_{0}.json" -f $ds)) }
  $aggCode = Invoke-Py (@("-u","scripts/aggr_eval.py","--ins") + $ins + @("--out",$Agg)) "aggr"
  Log "[aggr] end exit=$aggCode"
  if (-not $SkipPytest) {
    $pytestCode = Invoke-Py @("-u","-m","pytest","tests/numerical","--ignore=tests/numerical/test_gdn_oracle.py","-q") "pytest"
    Log "[pytest] end exit=$pytestCode"
  }
} elseif ($missing.Count -gt 0) {
  Log ("[bilan] datasets manquants : {0}" -f ($missing -join ", "))
}

# --- post-mortem (best effort) ---
try {
  $events = Get-WinEvent -FilterHashtable @{LogName=@('System','Application'); StartTime=$startTime} -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match 'python|powershell|nvlddmkm|WER|Application Error|Application Hang' } |
    Select-Object -First 40
  $evLog = Join-Path $OutDir ("events_chain_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
  if ($events) {
    $events | ForEach-Object { "{0} [{1}] {2}" -f $_.TimeCreated.ToString("yyyy-MM-dd HH:mm:ss"), $_.ProviderName, $_.Message } | Set-Content -Path $evLog
  } else {
    "aucun evenement correspondant" | Set-Content -Path $evLog
  }
} catch { Log "[postmortem] echec lecture evenements : $_" }

Set-Content -Path $dsFile -Value "done"
Stop-Job $hb -ErrorAction SilentlyContinue
Remove-Job $hb -Force -ErrorAction SilentlyContinue

$stateObj = [ordered]@{
  start = $startTime.ToString("yyyy-MM-dd HH:mm:ss")
  end = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  ckpt = $Ckpt
  datasets = $state
  missing = $missing
  dryrun = [bool]$DryRun
}
$stateObj | ConvertTo-Json -Depth 4 | Set-Content -Path $stateFile

if ($DryRun) { Log "[end] dryrun ok"; exit 0 }
if ($missing.Count -gt 0) { Log ("[end] incomplet, relance requise (manquants : {0})" -f ($missing -join ", ")); exit 2 }
Log "[end] chaine complete"
exit 0
