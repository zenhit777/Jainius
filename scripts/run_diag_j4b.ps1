# run_diag_j4b.ps1 -- diagnostic court "boucles vs lr de sauvegarde" (J4-bis, 17/10).
#
# Objectif (audit J4-bis 14, hypothese H1) : les ckpts de la phase 2
# degenerent-ils en boucles parce qu'ils ont ete sauves en plein regime
# (aucun n'est anneale : lr 3,4-9,5e-06 au pas sauve) plutot qu'en raison
# d'un effet cumulatif des prompts formates ? Reference : step_000699 (J4,
# lr 1,0e-06 a la sauvegarde).
#
# Methode : gsm8k --limit 6, T=0, meme protocole P-G/1.0 que la chaine
# d'eval (scripts/eval_j4.py) ; un JSON par ckpt ->
# _artefacts/j4b/diag/diag_<tag>_gsm8k.json.
#
# Concu pour etre lance DETACHE du shell de l'agent (AGENTS.md 11.1) :
# tache planifiee Windows "Jainius-DiagJ4bis".
#
# Garanties : heartbeat 60 s (PID + GPU + ckpt courant), idempotence
# (JSON valide = saute), faulthandler, stdout/stderr par tentative,
# post-mortem evenements Windows.
#
# Codes de sortie : 0 = complet ; 2 = ckpts manquants (la tache relance) ;
# 3 = erreur fatale.
#
# Controle utilisateur (GPU) :
#   arret  : Disable-ScheduledTask -TaskName Jainius-DiagJ4bis
#            puis Stop-ScheduledTask -TaskName Jainius-DiagJ4bis
#            (verifier qu'aucun python du run ne subsiste : Get-Process python)
#   reprise: Enable-ScheduledTask -TaskName Jainius-DiagJ4bis
#            puis Start-ScheduledTask -TaskName Jainius-DiagJ4bis
#            (saute les ckpts deja faits)

param(
  [string]$OutDir = "_artefacts/j4b/diag",
  [int]$Limit = 6,
  [int]$HeartbeatSeconds = 60,
  [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

# ckpts : tag -> chemin, dans l'ordre du diagnostic (lr de sauvegarde entre
# parentheses, cf. audit 14).
$ckpts = [ordered]@{
  "j4_699" = "_artefacts/j4/step_000699"            # 1,00e-06 (reference)
  "p2_749" = "_artefacts/j4b/phase2/step_000749"    # 9,51e-06
  "p2_799" = "_artefacts/j4b/phase2/step_000799"    # 7,93e-06
  "p2_849" = "_artefacts/j4b/phase2/step_000849"    # 5,67e-06
  "p2_899" = "_artefacts/j4b/phase2/step_000899"    # 3,36e-06 (le plus froid)
  "p2_949" = "_artefacts/j4b/phase2/step_000949"    # 6,02e-06
  "p2_952" = "_artefacts/j4b/phase2/step_000952"    # 5,57e-06
  "p2_979" = "_artefacts/j4b/phase2/step_000979"    # 5,84e-06
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDirAbs = (Resolve-Path $OutDir).Path
$diagLog = Join-Path $OutDirAbs "diag.log"
$hbLog = Join-Path $OutDirAbs "heartbeat.log"
$stateFile = Join-Path $OutDirAbs "diag_state.json"
$curFile = Join-Path $OutDirAbs "current_ckpt.txt"

function Log([string]$msg) {
  $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $diagLog -Value $line
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

# --- verification des ckpts ---
foreach ($tag in $ckpts.Keys) {
  if (-not (Test-Path $ckpts[$tag])) {
    Log ("[FATAL] ckpt manquant : {0} -> {1}" -f $tag, $ckpts[$tag])
    exit 3
  }
}

$gpu = ((nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -join " ")
$apps = ((nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) -join "; ")
Log ("[start] ckpts={0} limit={1} python={2} gpu={3} compute_apps=[{4}] dryrun={5}" -f $ckpts.Count, $Limit, $py, $gpu, $apps, $DryRun)

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
    Add-Content -Path $HbFile -Value ("{0} | alive | python={1} | gpu={2} | ckpt={3}" -f $ts, $pids, $g, $ds)
    Start-Sleep -Seconds $Seconds
  }
}
$hb = Start-Job -ScriptBlock $hbScript -ArgumentList $hbLog, $curFile, $repo, $HeartbeatSeconds, $supervisorPid

# --- fonctions d'execution ---
function Test-JsonOk([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  try { $null = Get-Content $path -Raw | ConvertFrom-Json; return $true } catch { return $false }
}

function Stop-Hb() {
  Stop-Job $hb -ErrorAction SilentlyContinue
  Remove-Job $hb -Force -ErrorAction SilentlyContinue
}

# --- arret rapide si tout est deja fait (relance AtLogOn) ---
if (-not $DryRun) {
  $allDone = $true
  foreach ($tag in $ckpts.Keys) {
    if (-not (Test-JsonOk (Join-Path $OutDirAbs ("diag_{0}_gsm8k.json" -f $tag)))) { $allDone = $false; break }
  }
  if ($allDone) {
    Log "[skip] diagnostic deja complet : rien a faire"
    Stop-Hb
    exit 0
  }
}

function Invoke-Py([string[]]$pyArgs, [string]$tag) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $outLog = Join-Path $OutDirAbs ("run_{0}_{1}.log" -f $tag, $ts)
  $errLog = Join-Path $OutDirAbs ("run_{0}_{1}.err" -f $tag, $ts)
  if ($DryRun) {
    Add-Content -Path $outLog -Value "[dryrun] would run: $py $($pyArgs -join ' ')"
    Start-Sleep -Seconds 1
    return 0
  }
  $env:PYTHONFAULTHANDLER = "1"
  $env:PYTHONUNBUFFERED = "1"
  $p = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $repo `
    -NoNewWindow -PassThru -Wait -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  return $p.ExitCode
}

# --- boucle ckpts (idempotente) ---
$state = [ordered]@{}
foreach ($tag in $ckpts.Keys) {
  $outJson = Join-Path $OutDirAbs ("diag_{0}_gsm8k.json" -f $tag)
  if (Test-JsonOk $outJson) {
    Log "[skip] $tag : resultat deja present et valide"
    $state[$tag] = "skip-existing"
    continue
  }
  if (Test-Path $outJson) {
    $bad = "$outJson.corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Move-Item -Force $outJson $bad
    Log "[warn] $tag : JSON illisible, deplace vers $bad"
  }
  Set-Content -Path $curFile -Value $tag
  $gpuNow = ((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join " ")
  Log "[ckpt:$tag] start (gpu_mem=$gpuNow MiB)"
  $t0 = Get-Date
  $code = Invoke-Py @("-X","faulthandler","-u","scripts/eval_j4.py","--ckpt",$ckpts[$tag],"--out",$outJson,"--datasets","gsm8k","--limit","$Limit") $tag
  $dur = [int]((Get-Date) - $t0).TotalSeconds
  Log "[ckpt:$tag] end exit=$code duree=${dur}s"
  $state[$tag] = "exit=$code"
}
Set-Content -Path $curFile -Value "post-ckpts"

# --- bilan ---
$missing = @()
foreach ($tag in $ckpts.Keys) {
  $outJson = Join-Path $OutDirAbs ("diag_{0}_gsm8k.json" -f $tag)
  if (-not (Test-JsonOk $outJson)) { $missing += $tag }
}
if ($missing.Count -gt 0) {
  Log ("[bilan] ckpts manquants : {0}" -f ($missing -join ", "))
}

# --- post-mortem (best effort) ---
try {
  $events = Get-WinEvent -FilterHashtable @{LogName=@('System','Application'); StartTime=$startTime} -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match 'python|powershell|nvlddmkm|WER|Application Error|Application Hang' } |
    Select-Object -First 40
  $evLog = Join-Path $OutDirAbs ("events_diag_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
  if ($events) {
    $events | ForEach-Object { "{0} [{1}] {2}" -f $_.TimeCreated.ToString("yyyy-MM-dd HH:mm:ss"), $_.ProviderName, $_.Message } | Set-Content -Path $evLog
  } else {
    "aucun evenement correspondant" | Set-Content -Path $evLog
  }
} catch { Log "[postmortem] echec lecture evenements : $_" }

Set-Content -Path $curFile -Value "done"
Stop-Hb

$stateObj = [ordered]@{
  start = $startTime.ToString("yyyy-MM-dd HH:mm:ss")
  end = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  limit = $Limit
  ckpts = $ckpts.Count
  results = $state
  missing = $missing
  dryrun = [bool]$DryRun
}
$stateObj | ConvertTo-Json -Depth 4 | Set-Content -Path $stateFile

if ($DryRun) { Log "[end] dryrun ok"; exit 0 }
if ($missing.Count -gt 0) { Log ("[end] incomplet, relance requise (manquants : {0})" -f ($missing -join ", ")); exit 2 }
Log "[end] diagnostic complet"
exit 0
