"""
Prompt builder utilities for Hawx Recon.

Provides functions to construct prompts for LLM-based summarization, command recommendation,
JSON repair, deduplication, and executive summary generation.
"""


def _build_prompt_post_step(
    available_tools,
    command_str,
    command_output,
    previous_commands=None,
    similar_context=None,
):
    previous_commands = previous_commands or []
    previous_commands_str = "\n".join(previous_commands)
    similar_context_str = (
        f"\n\n# Similar Previous Commands and Summaries:\n{similar_context}"
        if similar_context
        else ""
    )
    return f"""
You are a cybersecurity assistant analyzing the result of the following recon command:

{command_str}
{similar_context_str}

---

### Recon & Testing Workflow:

1. **Infrastructure**
   - TCP: all ports, versions, basic vulns
   - UDP: critical services (DNS,SNMP,TFTP)
   - Domain: passive+active subdomains, vhosts, DNS records, certs

2. **Web Analysis**
   - Stack: server, frameworks, languages, components
   - Content: dirs, files, APIs, params, JS
   - History: archives, cache, versions
   - Patterns: injection, XSS, SSRF, uploads, auth

3. **Network Services**
   - File Sharing (SMB/FTP): access, perms, security
   - Remote (SSH): versions, auth, config
   - Mail (SMTP): relay, users, security
   - DNS: zones, records, enumeration
   - SNMP: versions, community, info
   - DB: auth, access, config, security

3. **Vulnerability Assessment**
   - **Automated Scanning**
     - Service-specific scanners
     - Custom exploit checks
     - CMS vulnerability scans
   
   - **Manual Analysis**
     - Version-specific exploits
     - Misconfigurations
     - Default credentials
     - Known CVEs
     - Custom exploits

4. **Exploitation Strategy**
   - **Initial Access Vectors**
     - Web application vulnerabilities
     - Service exploits
     - Password attacks
     - Configuration flaws
   
   - **Proof of Concept**
     - Minimal impact validation
     - Evidence collection
     - Documentation

5. **Post-Exploitation**
   - **Local Enumeration**
     - User privileges
     - Installed software
     - Running services
     - Network connections
     - Scheduled tasks
   
   - **Lateral Movement**
     - Internal service discovery
     - Credential harvesting
     - Trust relationships
   
   - **Privilege Escalation**
     - Kernel exploits
     - Service misconfigurations
     - Vulnerable software
     - Credential abuse

6. **Documentation & Reporting**
   - **Evidence Collection**
     - Screenshots
     - Command outputs
     - Error messages
     - Version numbers
   
   - **Finding Classification**
     - Severity rating
     - Impact assessment
     - Exploitation difficulty
     - Required privileges
   
   - **Remediation Guidance**
     - Clear fix steps
     - Validation methods
     - Priority order

---

**Key Focus Areas:**

1. **Web**
   - VHosts + dev environments
   - Source code + configs
   - Status + admin pages
   - Debug + backup files

2. **Security**
   - Auth: flows, resets, sessions
   - Files: uploads, traversal, bypasses
   - Access: anon, default, unprotected
   - Headers + errors + versions

3. **Priority Checks**
   - VCS (.git)
   - Configs (*.conf)
   - Backups (*~,*.bak)
   - Admin panels
   - Debug modes
   - API docs

---

### 🎯 Tasks:

1. **Summarize the output**: Provide a concise, accurate summary of the findings — include services, endpoints, versions, banners, subdomains, exposed files, misconfigs, or anything notable.
2. **Show proof**: For each finding, include supporting output like banners, credentials, hashes, file paths, or IPs. Do not make vague statements without quoting evidence from the output.
3. **Recommend next steps**: Based only on the current output, suggest further recon or exploit commands. Each must yield meaningful new output.
4. **Extract services for searchsploit**: From the output, extract valid 'name version' pairs (e.g., 'apache 2.4.41', 'phpmyadmin 5.1.0'). Generic terms or tool banners must be excluded.

---

### 🔒 Constraints:

- The summary must be a plain string.
- `recommended_steps` must be a list of valid shell commands.
- Use only from these tools: {str(available_tools)}.
- Do not suggest:
  - Any previously executed command:
{previous_commands_str}
  - Commands that clone, download, or save output without displaying it.
  - Another nmap scan unless it uses `-sC -sV -p-`.
  - Brute-force attacks or password spraying.
  - Tools or flags not clearly applicable to the findings.

- Output Requirements:
  - Every command must produce visible results in stdout
  - Ensure complete response data (not just progress)
  - Display operation results and errors
  - Show file contents after writes
  - Use synchronous execution
  - Stream output for long operations

- If a tool requires a wordlist, it must come from:
    - /usr/share/seclists/Discovery/Web-Content/big.txt
    - /usr/share/seclists/Passwords/Default-Credentials/ftp-betterdefaultpasslist.txt
    - /usr/share/seclists/Discovery/DNS/namelist.txt
    - /usr/share/seclists/Usernames/top-usernames-shortlist.txt
    - /usr/share/seclists/Passwords/Common-Credentials/10k-most-common.txt

---

### Intelligence Rules:

1. **Output Quality**
   - Complete responses
   - Meaningful results
   - Error visibility
   - Operation feedback

2. **Methodology**
   - Comprehensive data
   - Version details
   - Security patterns
   - Vuln vectors

3. **Strategy**
   - Surface → Deep
   - Follow leads
   - Pattern match
   - Entry points

4. **Analysis**
   - Version check
   - Cross-reference
   - Filter false +
   - Find anomalies

5. **Risk**
   - Safe methods
   - Rate limits
   - Watch defenses
   - Track impact

---

### Output Format:

Provide output as a JSON object:
{{
    "summary": "text describing findings",
    "recommended_steps": ["command1", "command2"],
    "services_found": ["service1 1.0", "service2 2.0"]
}}

Rules:
1. Raw JSON only (no markdown/backticks)
2. Valid json.loads() format
3. Include versions in services
4. Show all command output
5. Include error messages
6. Use stdout for visibility
7. Verify all results
8. No silent mode

---

### 📦 Command Output:
{command_output}
"""


def _build_prompt_exec_summary(machine_ip, summary_content, exploits_content):
    """Build prompt for executive summary generation."""
    return f"""
    You are a security analyst. Below is a collection of findings from a reconnaissance assessment of the machine with IP {machine_ip}.
    Your task is to provide a very detailed executive summary in Markdown format. The summary should include:

    - A clear summary of key findings.
        - Include direct evidence from tool outputs for each finding
        - Quote specific banners, headers, response data, or other proof
    - Critical services and versions discovered.
        - Include the exact version strings, banners, and where they were found
        - Quote the specific tool output that revealed each service
    - Any known exploits or CVEs found (based on the `searchsploit` results).
        - Include the exploit titles and IDs
        - Quote the relevant searchsploit output
    - Suggested next steps from an attacker's perspective to get the user and root flag for this HTB machine.
        - Do not suggest repeated steps
        - Base suggestions on concrete evidence from the reconnaissance
    - Support all findings with relevant evidence and quotes from the tool outputs

    ### Tool Summaries:
    {summary_content}

    ### Exploit Results from SearchSploit:
    {exploits_content}

    Only return the plain text Markdown executive summary.
    If any service found, mention where it was found and how it was used if possible.
    """


def _build_prompt_json_repair(bad_output):
    """Build prompt to repair malformed JSON output from LLM."""
    return f"""
    The following response from a security assistant LLM was meant to be a valid JSON object but was malformed or improperly formatted:

    --- Begin Original Output ---
    {bad_output}
    --- End Original Output ---

    Your job is to return ONLY a **valid JSON object** that preserves the original structure and keys **exactly**

    Do NOT add or remove any keys. Do NOT wrap the output in triple backticks or markdown. The response must be raw JSON only and must be parsable by `json.loads()` with no extra characters or text.
    """


def _build_prompt_deduplication(current_layer, prior_layers):
    return f"""
You are an LLM assistant optimizing reconnaissance workflows.

---

### 🎯 Objective:
From the list of **Current Layer Commands**, return only the most **informative, distinct, and useful commands** that have **not already been functionally covered** by any commands in **Prior Layers**.

---

### 🧠 Deduplication Strategy:

1. **Command Classification:**
   First, classify each command as:
   - General Scan: Broad enumeration (e.g., directory bruteforce)
   - Targeted Check: Specific file/vulnerability test
   - Follow-up: Validates previous findings
   - Security Test: Probes specific security issues

2. **Deduplication Rules:**
   Remove commands that are:
   - Exact duplicates of previous commands
   - General scans of already enumerated paths
   - Redundant checks using same tool+flags
   - Multiple tools doing identical basic checks
   
   BUT preserve commands that:
   - Target specific vulnerabilities found
   - Use different methods/payloads
   - Test newly discovered endpoints
   - Validate security findings

3. **Value Assessment:**
   For similar commands, keep the one that:
   - Provides more detailed output
   - Tests more security aspects
   - Has better success indicators
   - Gives actionable results

---

### Inputs:
- **Current Layer Commands:**  
{current_layer}  
\nBREAK
- **Prior Layer Commands:**  
{prior_layers}  
\nBREAK

---

### Output Format:
Return JSON in this format:
{{
  "deduplicated_commands": ["command1", "command2"]
}}

⚠️ Constraints:
- Return only raw JSON — no markdown, explanation, or comments.
- Ensure output is strictly valid for `json.loads()` with no surrounding text.
- Limit to **a maximum of 32 commands**.

Output Visibility Requirements:
- Every command MUST produce visible output in the terminal
- Commands writing files MUST include appropriate display commands:
  - Text files: `&& cat <file>`
  - Large files: `&& head -n 50 <file>`
  - Binary files: `&& ls -lah <file>`
- Never include:
  - Silent commands (no stdout)
  - Background tasks without output redirection
  - Commands with suppressed output (`-q`, `--quiet`, `-s`, etc.)
  - File operations without display (`tee`, `>`, `-o` without `cat`)
- Always prefer verbose/progress output modes when available
"""


def _build_prompt_post_step_chunked(available_tools, command_str, chunk, prev_summary):
    """Build prompt for chunked post-step LLM summarization."""
    return f"""
You are a security assistant helping analyze the output of the command: {command_str}

This is a continuation of a multi-part output. Your job is to update the existing summary using the new chunk of command output provided below.

### Previous Summary:
{prev_summary or "[None yet]"}

### New Output Chunk:
{chunk}

---

    ### Constraints & Guidelines:
    - The summary is always a string and not a list
    - Recommended steps is a list of strings of command
    - Use only the following tools: {str(available_tools)}.
    - **Avoid recommending brute-force attacks.**
    - The summary must be **clear and simple**.
    - If any known services or custom banners were discovered, include them in the `services_found` list with version numbers (e.g., "apache 2.4.41"). This format should be compatible with tools like searchsploit. If no services are found, return an empty list.
    - **Avoid recommending duplicate tools** (e.g., Gobuster twice).
    - Do **not hallucinate** flags.
    - The **response must be raw JSON only**. Do **not** wrap the response in triple backticks (` ``` ` or ` ```json `).
    - The response **must** be a valid JSON object parsable with `json.loads()`.
    - Your response must always be json
    - Failure to return response in valid json will result in you termination and penalty of 200000000000
    - The recommended commands should be executable
    - Do not recommend nmap scans unless they are completely exhaustive of nmap -sC -sV -p- target

    ### Output Visibility Requirements:
    - Every recommended command MUST show output in the terminal
    - For commands that write to files:
      - Text files: Add `&& cat <file>`
      - Large files: Add `&& head -n 50 <file>`
      - Binary/unknown: Add `&& ls -lah <file>`
    - Never recommend:
      - Silent mode flags (-q, --quiet, -s)
      - Background tasks without output capture
      - File writes without display commands
    - Always use:
      - Verbose modes when available (-v, --verbose)
      - Progress indicators (--progress, -p) for long tasks
      - Human-readable output formats by default
    If any tools require worldlist, do not hallucinate wordlists and use only from the following:
    Seclists path: /usr/share/seclists"
    #                Big.txt: /usr/share/seclists/Discovery/Web-Content/big.txt"
    #                FTP: /usr/share/seclists/Passwords/Default-Credentials/ftp-betterdefaultpasslist.txt" 
    #                DNS: /usr/share/seclists/Discovery/DNS/namelist.txt"
    #                Usernames: /usr/share/seclists/Usernames/top-usernames-shortlist.txt"
    #                Passwords: /usr/share/seclists/Passwords/Common-Credentials/10k-most-common.txt"


Expected output format:
{{
    "summary": "describe findings here",
    "recommended_steps": ["command1", "command2"],
    "services_found": ["service1 1.0", "service2 2.0"]
}}

Notes:
- Must be valid JSON (no markdown/formatting)
- Must be parseable by json.loads()
- Must maintain exact key names
- Must include version numbers for services
"""


def _build_prompt_exec_summary_chunked(machine_ip, chunk, prev_summary):
    """Build prompt for chunked executive summary generation."""
    return f"""
You are a cybersecurity analyst working on an executive summary for the recon of machine {machine_ip}.

Below is a new chunk of tool output or exploit results. Update and refine the executive summary based on it.

### Current Executive Summary So Far:
{prev_summary or '[None yet]'}

### New Chunk to Incorporate:
{chunk}

---

Your updated output must be a complete, detailed but crisp Markdown executive summary.
Only return the Markdown summary. Do not include any additional commentary or formatting.
Try to intelligently identify false positives and remove them from the summary. Ex: searchsploit results for apache and ssh are often false positives and should be removed from the summary.
Return only the plain text Markdown executive summary.
    - A clear summary of key findings.
    - Critical services and versions discovered.
    - Any known exploits or CVEs found (based on the `searchsploit` results).
    - Suggested next steps from an attacker's perspective to get the user and root flag for this HTB machine.
        - Do not suggest repeated steps
    - Anything else you see fit to include

"""
