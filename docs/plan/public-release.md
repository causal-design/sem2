# Proposal: make SEM2 public

**Status:** proposed for team discussion; not authorization to publish.<br>
**Date:** 14 October 2026<br>
**Preferred code license:** GNU GPL version 3, subject to team and copyright-holder approval.

## Decision requested

Make `causal-design/sem2` public only after removing meeting transcripts from its Git history and agreeing on licensing and publication rights. Keep the existing repository rather than creating a new public repository, provided we can resolve the remaining GitHub exposure risks.

This proposal preserves the meeting notes, summaries, implementation plans, and code development history. It does not propose deleting all of `docs/meetings/` or starting a new Git history.

Merging this PR records the agreed plan. It does **not** delete transcripts, rewrite history, add a license, or change repository visibility. Those actions require a separate, coordinated execution after the approval gates below are satisfied.

## Transcript removal scope

Remove only the meeting transcripts under `docs/meetings/`, recursively, including their earlier versions and historical paths. The current local checkout contains these five transcript files:

| Current path | Planned treatment |
|---|---|
| `docs/meetings/2026-09-03-and-07/transcript-f01-f09.txt` | Remove from all relevant history |
| `docs/meetings/2026-09-03-and-07/transcript-f12-f27.txt` | Remove from all relevant history |
| `docs/meetings/2026-09-03-and-07/transcript-f32-f55.txt` | Remove from all relevant history |
| `docs/meetings/2026-09-30-transcript.txt` | Remove from all relevant history |
| `docs/meetings/2026-10-08-transcript.txt` | Remove from all relevant history |

The locally available history also records these earlier locations:

- `docs/meetings/2026-09-03/transcript-f12-f27.txt`
- `docs/meetings/2026-09-03/transcript-f32-f55.txt`
- `docs/meetings/2026-09-07/transcript-f01-f09.txt`

This is an initial inventory, not an exhaustive audit of GitHub branches, tags, PR references, renamed files outside this folder, or copies of transcript contents. Finalize the removal list against a fresh, complete view of the repository before rewriting. Filename matching alone cannot establish that all confidential material has been found.

Keep non-transcript documents, including `*-meeting.md`, `summary-*.md`, reviews, and product/implementation plans, subject to a confidentiality review. Check those retained documents and other publication surfaces for copied transcript excerpts, personal information, and confidential discussions. Repair links to removed transcripts without copying their contents into replacement documents or PR comments.

## Why an ordinary deletion PR is insufficient

Deleting a transcript in a new commit leaves earlier versions readable in older commits. Removing it from older commits requires rewriting history, for example with `git-filter-repo`, then replacing the affected remote branches and tags through coordinated force-pushes.

The rewritten history retains other files and their changes. Mixed code/transcript commits retain their non-transcript changes; transcript-only commits generally disappear if they become empty. Rewriting can also simplify merge history. Affected commits and their descendants get new commit IDs, and commit/tag signatures can be removed. Existing commit links, PR diffs, and review comments may be affected.

A PR cannot rewrite the existing history by being merged. It can document agreement and the execution plan, which is the purpose of this proposal.

## Coordinated cleanup and publication plan

Keep the repository private throughout preparation and cleanup.

1. **Agree on scope and ownership.** Confirm the transcript removal list, which records should remain public, the license decision, and who will execute and independently verify the cleanup. Merge or close affected open PRs, or explicitly plan how to recreate them.
2. **Pause pushes and make a private backup.** Preserve outstanding contributor work and an access-controlled backup of the original repository. Do not retain the unfiltered backup as a branch or tag in the repository that will become public. Treat the backup as confidential and agree on its retention and access policy.
3. **Inventory all relevant refs and exposure surfaces.** Inspect branches, tags, historical paths, PR references, forks, and any transcript-bearing attachments, releases, Actions artifacts, or other GitHub content. A Git history filter does not clean those other surfaces.
4. **Rewrite in a fresh, dedicated clone.** Use `git-filter-repo` to remove the approved transcript paths, including former locations, while preserving non-transcript files. Do not perform the rewrite in a contributor's working checkout. Save the rewrite report and commit mapping privately for verification and support requests.
5. **Verify before pushing.** Compare the resulting tree against the expected transcript-only removal; check all relevant rewritten branches and tags for removed paths and known transcript content or blob IDs. Confirm retained notes and plans remain, review changed commit/merge structure, and verify retained repository-relative links. Remove any separately identified confidential material before proceeding.
6. **Replace the remote history.** Coordinate temporary changes to protections or rulesets as needed, push the approved rewritten refs, restore protections, and confirm no branch/tag update failed. Do not use a blanket mirror push without reviewing its ref-deletion effects. Keep the repository private.
7. **Resolve old GitHub references and copies.** A force-push alone may leave original commits accessible through PR references, cached pages, or direct commit IDs. Request GitHub Support cleanup where eligible, including garbage collection and cached views. Coordinate with fork owners and collaborators; we cannot erase other people's downloaded copies.
8. **Re-clone and prevent reintroduction.** After preserving outstanding work safely, collaborators should use fresh clones. Do not merge old branches or push old refs into the cleaned repository. Agree on an approved private location for future transcripts and add scoped ignore/pre-commit safeguards as a follow-up; `.gitignore` does not remove existing history.
9. **Complete licensing and publication review.** Add the approved license and metadata through a separate normal PR, verify code/content rights, inspect the remote again, and obtain explicit approval to change visibility. Only then make the repository public.

GitHub Support does not promise to remove all unwanted content: its policy limits assistance to qualifying sensitive data, and it will not remove non-sensitive data merely on request. If old transcript-bearing references remain accessible or cleanup eligibility is uncertain, **do not make this repository public**. Reconsider a new, independent repository containing verified cleaned history while retaining the original privately, or postpone publication. Do not promise complete erasure from backups, clones, or forks.

## License proposal and alternatives

GitHub permits a public repository without a license, but public visibility alone does not grant the normal rights to reuse, modify, and redistribute code. Use a standard open-source license rather than a custom restriction.

The preference expressed in this discussion is **GPLv3**. It suits a goal of keeping distributed derivative software open source rather than allowing proprietary derivatives. All listed options permit commercial use; GPL is not a noncommercial license.

| Option | Main effect | Relevant tradeoff |
|---|---|---|
| GPLv3 | Strong copyleft: distributed covered derivatives must preserve GPL freedoms and provide corresponding source as required | Limits integration into distributed closed-source combined software; includes an explicit contributor patent license |
| GPLv2-only | Strong copyleft under version 2 | Not compatible with Apache-2.0 for a combined distributed work; lacks GPLv3's express contributor patent-license provisions |
| GPLv2-or-later | Recipients may choose version 2 or a later GPL version | Allows use under GPLv3; this is lavaan's choice, not a GPLv2-only restriction |
| Apache-2.0 | Permissive reuse, including proprietary derivatives, with notices and modification notices | Explicit contributor patent grant, but no requirement for distributed derivatives to remain open source |
| MIT | Permissive reuse, including proprietary derivatives, with copyright/license notices | Shorter terms; no express patent grant |

For a new copyleft project, GPLv3 is preferred over GPLv2-only because of its express patent terms, Apache-2.0 compatibility, and clearer compliance provisions. The exact declaration remains a team decision: `GPL-3.0-only` fixes version 3; `GPL-3.0-or-later` lets recipients choose future GPL versions too. Do not silently treat these as interchangeable.

GPL does not require publishing private/internal modifications or submitting changes upstream. Merely running modified GPL software as a hosted service generally does not trigger source-sharing obligations; AGPL addresses network use separately and is not the current proposal. Using SEM2 for research does not, by itself, require licensing datasets, papers, or ordinary numerical results under GPL.

### Other open-source SEM software

The current published package declarations/license files checked for this discussion show a mix, not a single SEM-community standard:

| Project | Language | Declared license |
|---|---|---|
| [lavaan](https://cran.r-project.org/web/packages/lavaan/DESCRIPTION) | R | GPL version 2 or later |
| [sem](https://cran.r-project.org/web/packages/sem/DESCRIPTION) | R | GPL version 2 or later |
| [piecewiseSEM](https://cran.r-project.org/web/packages/piecewiseSEM/DESCRIPTION) | R | GPL version 3 |
| [OpenMx](https://cran.r-project.org/web/packages/OpenMx/DESCRIPTION) | R / C++ | Apache-2.0 |
| [semopy](https://gitlab.com/georgy.m/semopy/-/raw/master/LICENSE) | Python | MIT |

Implementing similar statistical methods does not automatically require adopting another project's license. Copying or porting its implementation code can carry its license obligations. Audit provenance, bundled code, and dependencies before approving SEM2's license; no complete license audit has been performed for this proposal.

## Ownership, patents, and future licensing

The team reports that it developed all SEM2 code and owns it. Confirm legal ownership and authorization with the relevant contributors and, where applicable, universities or employers. Authorship alone does not establish ownership. Papers, reports, datasets, and other third-party material may require separate permissions or license terms; a software license does not automatically clear them for publication.

An explicit contributor patent grant gives users royalty-free permission to exercise covered patent claims that contributors have authority to license. It does not transfer patent ownership, cover every contributor patent, or protect against unrelated third-party patents. GPLv3, like Apache-2.0, contains an express contributor patent license. Choosing GPLv2 or MIT is not a dependable way to preserve exclusive patent commercialization rights.

Neither Apache-2.0 nor GPLv3 prohibits applying for or owning patents. However, covered patent permissions already granted cannot simply be withdrawn through a license change. Whether a later-granted patent falls within an existing grant needs legal analysis.

Separately, public disclosure of an invention through code, papers, or presentations can jeopardize patent eligibility regardless of the software license. Grace periods and software-invention eligibility vary by jurisdiction. If preserving patent options matters, consult the institution's IP/technology-transfer office **before** public release or licensing and assess any earlier disclosures.

### Could later SEM2 versions become closed source?

Yes, if the relevant copyright holders own or have adequate relicensing rights to all included code and approve the change. They can offer their code under GPLv3 now and different terms later, including parallel GPL/commercial licensing. Merely replacing `LICENSE` is not sufficient when other people's code is involved.

- Previously released GPL versions remain usable, modifiable, and redistributable under GPL, including through independent forks. Compliant recipients' existing permissions cannot be revoked.
- Future outside contributions accepted under GPL alone do not automatically come with proprietary relicensing permission. If this option is important, agree on contribution terms in advance, potentially through a lawyer-reviewed agreement, or obtain additional permission later.
- Third-party GPL code cannot automatically be relicensed just because SEM2's own code belongs to the team.
- Future license changes do not undo patent permissions already granted.

This is a planning summary, not a legal opinion. Institutional ownership, patent strategy, and any proposed contributor/relicensing agreement need appropriate legal review.

## Approval gates

Team review should resolve these decisions explicitly; unchecked items block the affected execution step or public release:

- [ ] Approve the transcript-only removal scope, including all discovered historical paths.
- [ ] Approve retained notes, summaries, reviews, plans, reports, and data for public disclosure after confidentiality review.
- [ ] Confirm copyright ownership, publication authority, dependency/code provenance, and separate content permissions.
- [ ] Approve GPLv3 and choose `GPL-3.0-only` or `GPL-3.0-or-later`.
- [ ] Resolve patent concerns before release or licensing, including earlier public disclosures.
- [ ] Decide whether future proprietary licensing matters and whether contribution terms should preserve that option.
- [ ] Assign cleanup and independent verification owners; agree on the push freeze, secure backup, and collaborator migration.
- [ ] Verify rewritten local and remote history, restored protections, and cleaned GitHub exposure surfaces; resolve support/fork limitations.
- [ ] Merge a follow-up licensing PR with the approved `LICENSE`, README statement, package metadata, and clear content scope.
- [ ] Obtain explicit final approval to make the repository public. Approval of this proposal alone is not that approval.

## References

- [GitHub: removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [GitHub: licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [GNU: guide to GPLv3 and differences from GPLv2](https://www.gnu.org/licenses/quick-guide-gplv3.html)
- [GNU GPLv3 license text, including the patent provisions in section 11](https://www.gnu.org/licenses/gpl-3.0.html)
- [GNU FAQ: private modifications and source disclosure](https://www.gnu.org/licenses/gpl-faq.html#GPLRequireSourcePostedPublic)
- [GNU FAQ: offering an author's code under multiple licenses](https://www.gnu.org/licenses/gpl-faq.html#ReleaseUnderGPLAndNF)
- [GNU FAQ: permission needed to relicense contributions](https://www.gnu.org/licenses/gpl-faq.html#Consider)
- [Apache-2.0 license text, including the patent grant in section 3](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT license text and summary](https://choosealicense.com/licenses/mit/)
- [Open Source Guides: ownership, licensing, and contributor agreements](https://opensource.guide/legal/)
- [WIPO: patent eligibility and disclosure before filing](https://www.wipo.int/en/web/patents/faq_patents#confidentiality)
