# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-20

### Breaking Changes

**Renamed properties for consistency:**
- `CVDVulnerability.is_kev` → `CVDVulnerability.kev` (matches pattern of `epss`, `cvss_score`)
- `CVDArray.has_public_exploit` → `CVDArray.is_weaponized` (matches CVDVulnerability naming)
- `CVDArray.premature_disclosure` → `CVDArray.is_premature_disclosure` (adds "is_" prefix)

**Removed deprecated properties:**
- `CVDArray.cubes` → Use `CVDArray.fix_path`
- `CVDArray.event_timestamps` → Use `CVDArray.events` (new) or exploded properties (`V_timestamps`, etc.)

**Removed internal properties:**
- `CVDVulnerability.state_encoded` → Internal detail, use `state` property
- `CVDVulnerability.state_int` → Internal detail, use `state` property

### Added

**New CVDArray properties for API parity:**
- `CVDArray.cve_ids` - Direct access to CVE identifiers array (primary user-facing identifier)
- `CVDArray.events` - Dict-style timestamp access matching `CVDVulnerability.events` pattern
- `CVDArray.is_zero_day_exploit` - Exploit public before vendor awareness
- `CVDArray.is_zero_day_attack` - Attacks before vendor awareness
- `CVDArray.is_coordinated` - Vendor aware before public disclosure
- `CVDArray.is_responsible_disclosure` - V→F→P ordering maintained
- `CVDArray.has_fix_before_exploit` - Fix ready before exploit public
- `CVDArray.has_fix_before_attack` - Fix ready before attacks observed
- `CVDArray.has_deployment_before_exploit` - Fix deployed before exploit public
- `CVDArray.has_deployment_before_attack` - Fix deployed before attacks observed
- `CVDArray.is_private_attack` - Attacks without public exploit
- `CVDArray.is_mass_exploitation` - Both exploit public and attacks observed

**DataFrame export enhancements:**
- Added 10 analytical properties to `to_dataframe()`: is_zero_day_exploit, is_zero_day_attack, is_coordinated, is_responsible_disclosure, has_fix_before_exploit, has_fix_before_attack, has_deployment_before_exploit, has_deployment_before_attack, is_private_attack, is_mass_exploitation
- Added 8 CVSS 3.1 base metrics: attack_vector, attack_complexity, privileges_required, user_interaction, scope, confidentiality_impact, integrity_impact, availability_impact
- Added state_label property (compressed state labels like 'VFP')
- DataFrame now exports ~48 columns with analytics enabled (19 base + 10 new analytics + 8 existing analytics + 8 CVSS metrics + 1 state label + timestamps)

### Changed

**Documentation improvements:**
- Standardized property documentation style across CVDVulnerability and CVDArray
- Complex properties now have detailed 3-line docs with computation details
- Simple properties kept concise with 1-line descriptions

### Migration Guide

#### Renamed properties

**CVDVulnerability.is_kev → kev:**
```python
# OLD
vuln.is_kev = True
if vuln.is_kev:
    print("KEV vulnerability")

# NEW
vuln.kev = True
if vuln.kev:
    print("KEV vulnerability")
```

**CVDArray.has_public_exploit → is_weaponized:**
```python
# OLD
mask = arr.has_public_exploit
weaponized = arr[mask]

# NEW
mask = arr.is_weaponized
weaponized = arr[mask]
```

**CVDArray.premature_disclosure → is_premature_disclosure:**
```python
# OLD
mask = arr.premature_disclosure
premature = arr[mask]

# NEW
mask = arr.is_premature_disclosure
premature = arr[mask]
```

#### Removed deprecated properties

**arr.cubes → arr.fix_path:**
```python
# OLD (deprecated alias)
cubes = arr.cubes

# NEW (correct name)
fix_path = arr.fix_path
```

**arr.event_timestamps → arr.events or exploded properties:**
```python
# OLD (deprecated dict)
v_times = arr.event_timestamps[CVDEvent.V]

# NEW (option 1 - dict-style, matches vuln.events)
v_times = arr.events[CVDEvent.V]

# NEW (option 2 - exploded properties, more efficient)
v_times = arr.V_timestamps
```

#### Removed internal properties

**vuln.state_encoded / vuln.state_int → vuln.state:**
```python
# OLD (removed - internal implementation details)
state_int = vuln.state_encoded
state_int = vuln.state_int

# NEW (use string representation)
state_str = vuln.state  # Returns 'VFdPxa' format
```

#### New capabilities

**Direct CVE ID access:**
```python
# NEW - Primary identifier for filtering
arr.cve_ids  # Returns array of CVE ID strings

# Filter by CVE ID
mask = arr.cve_ids == 'CVE-2024-001'
critical = arr[mask]

# Find multiple CVEs
target_cves = ['CVE-2024-001', 'CVE-2024-002', 'CVE-2024-003']
mask = np.isin(arr.cve_ids, target_cves)
selected = arr[mask]
```

**Consistent timestamp API:**
```python
# NEW - Dict-style access (matches CVDVulnerability.events)
arr.events[CVDEvent.V]  # Same as arr.V_timestamps
arr.events[CVDEvent.F]  # Same as arr.F_timestamps

# Useful for dynamic event access
for event in [CVDEvent.V, CVDEvent.F, CVDEvent.P]:
    timestamps = arr.events[event]
    print(f"{event.name}: {timestamps}")
```

**Advanced analytics (10 new properties):**
```python
# NEW - Zero-day detection
arr.is_zero_day_exploit  # X before V
arr.is_zero_day_attack   # A before V

# NEW - Coordination quality
arr.is_coordinated             # V before P
arr.is_responsible_disclosure  # V→F→P ordering

# NEW - Fix timing
arr.has_fix_before_exploit  # F before X
arr.has_fix_before_attack   # F before A

# NEW - Deployment timing
arr.has_deployment_before_exploit  # D before X
arr.has_deployment_before_attack   # D before A

# NEW - Threat characteristics
arr.is_private_attack      # A without X (targeted attacks)
arr.is_mass_exploitation   # X and A both (widespread)

# Example: Find high-priority vulnerabilities
high_priority = arr[
    arr.is_zero_day_exploit |
    arr.is_mass_exploitation |
    (arr.kev & ~arr.has_fix_before_exploit)
]
```

## [0.1.0] - 2026-01-13

### Added

- Initial release with CVD state machine implementation
- Support for NVD 1.1 and 2.0 formats
- Batch operations with numpy
- Analytics and desiderata scoring
- EPSS and KEV integration
- DataFrame export functionality
