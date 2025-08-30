param(
  [Parameter(Mandatory=$true)][string]$Repo  # format: owner/repo
)

# Requires: GitHub CLI (gh), PowerShell 7+
# Usage: pwsh scripts/bootstrap-issues.ps1 -Repo "owner/repo"

function Ensure-GhAuth {
  try {
    gh auth status 1>$null 2>$null
  } catch {
    Write-Error "GitHub CLI not authenticated. Run: gh auth login"
    exit 1
  }
}

function New-OrUpdate-Milestone {
  param([string]$Repo,[string]$Title,[string]$DueOn)
  # Try to find existing milestone by title
  $existing = gh api -X GET repos/$Repo/milestones --paginate | ConvertFrom-Json | Where-Object { $_.title -eq $Title }
  $isoDue = if ($DueOn -match 'T') { $DueOn } else { "$DueOnT00:00:00Z" }
  if ($existing) {
    Write-Output "Milestone exists: $Title (#$($existing.number))"
    return $existing
  }
  $payload = @{ title=$Title; due_on=$isoDue }
  $resp = gh api -X POST repos/$Repo/milestones -f title="$Title" -f due_on="$isoDue" | ConvertFrom-Json
  Write-Output "Created milestone: $Title (#$($resp.number))"
  return $resp
}

function New-OrUpdate-Label {
  param([string]$Repo,[string]$Name,[string]$Color)
  $nameEnc = [uri]::EscapeDataString($Name)
  $getUrl = "repos/$Repo/labels/$nameEnc"
  $exists = $null
  try { $exists = gh api $getUrl | ConvertFrom-Json } catch {}
  if ($exists) {
    gh api -X PATCH $getUrl -f new_name="$Name" -f color="$Color" 1>$null
    Write-Output "Updated label: $Name"
  } else {
    gh api -X POST repos/$Repo/labels -f name="$Name" -f color="$Color" 1>$null
    Write-Output "Created label: $Name"
  }
}

function Get-Milestone-Map {
  param([string]$Repo)
  $list = gh api -X GET repos/$Repo/milestones --paginate | ConvertFrom-Json
  $map = @{}
  foreach ($m in $list) { $map[$m.title] = $m.number }
  return $map
}

Ensure-GhAuth

# Load milestones and labels
$milestonesPath = "project-management/milestones.json"
$labelsPath = "project-management/labels.json"
$issuesPath = "project-management/issues.csv"

if (-not (Test-Path $milestonesPath)) { Write-Error "Missing $milestonesPath"; exit 1 }
if (-not (Test-Path $labelsPath)) { Write-Error "Missing $labelsPath"; exit 1 }
if (-not (Test-Path $issuesPath)) { Write-Error "Missing $issuesPath"; exit 1 }

$milestones = Get-Content $milestonesPath | ConvertFrom-Json
$labels = Get-Content $labelsPath | ConvertFrom-Json

Write-Output "Syncing milestones..."
foreach ($m in $milestones) { New-OrUpdate-Milestone -Repo $Repo -Title $m.title -DueOn $m.due_on | Out-Null }

Write-Output "Syncing labels..."
foreach ($l in $labels) { New-OrUpdate-Label -Repo $Repo -Name $l.name -Color $l.color }

# Refresh milestone map after potential creations
$msMap = Get-Milestone-Map -Repo $Repo

Write-Output "Creating issues..."
# Import CSV and create issues
# CSV headers: Title,Body,Labels,Milestone
$issues = Import-Csv -Path $issuesPath
foreach ($i in $issues) {
  $title = $i.Title
  $body = $i.Body
  $labelList = ($i.Labels -replace '\|', ',')
  $milestoneTitle = $i.Milestone
  $args = @('-R', $Repo, '-t', $title, '-b', $body)
  if ($labelList -and $labelList.Trim().Length -gt 0) { $args += @('-l', $labelList) }
  if ($milestoneTitle -and $milestoneTitle.Trim().Length -gt 0) { $args += @('-m', $milestoneTitle) }

  # Check if issue already exists by title (best-effort):
  $existing = gh issue list -R $Repo --search "in:title `"$title`"" --json number,title | ConvertFrom-Json | Where-Object { $_.title -eq $title }
  if ($existing) {
    Write-Output "Issue exists, skipping: $title (#$($existing.number))"
    continue
  }

  gh issue create @args | Out-Null
  Write-Output "Created issue: $title"
}

Write-Output "Done."

