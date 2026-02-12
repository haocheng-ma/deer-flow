// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { ChevronDown, Database, FileText, FolderUp, Upload } from "lucide-react";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { resolveServiceURL } from "~/core/api/resolve-service-url";
import type { Resource } from "~/core/messages";
import { cn } from "~/lib/utils";

import type { Tab } from "./types";

// 与 MinerU 支持类型对齐：.doc, .docx, .pdf, .ppt, .pptx
const EXT_RE = /\.(doc|docx|pdf|ppt|pptx)$/i;
const MAX_FILES_PER_REQUEST = 200;

async function readDirectory(reader: FileSystemDirectoryReader): Promise<File[]> {
  const allEntries: FileSystemEntry[] = [];
  let batch: FileSystemEntry[];
  do {
    batch = await new Promise((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    allEntries.push(...batch);
  } while (batch.length > 0);
  const files: File[] = [];
  for (const entry of allEntries) {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) =>
        (entry as FileSystemFileEntry).file(res, rej),
      );
      if (EXT_RE.test(file.name)) files.push(file);
    } else if (entry.isDirectory) {
      const subReader = (entry as FileSystemDirectoryEntry).createReader();
      const subFiles = await readDirectory(subReader);
      files.push(...subFiles);
    }
  }
  return files;
}

async function collectFilesFromDataTransfer(dataTransfer: DataTransfer): Promise<File[]> {
  const files: File[] = [];
  const items = Array.from(dataTransfer.items);
  for (const item of items) {
    if (item.kind !== "file") continue;
    const entry = "webkitGetAsEntry" in item ? item.webkitGetAsEntry() : null;
    if (!entry) {
      const file = item.getAsFile();
      if (file && EXT_RE.test(file.name)) files.push(file);
      continue;
    }
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) =>
        (entry as FileSystemFileEntry).file(res, rej),
      );
      if (EXT_RE.test(file.name)) files.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const dirFiles = await readDirectory(reader);
      files.push(...dirFiles);
    }
  }
  return files;
}

export const RAGTab: Tab = () => {
  const t = useTranslations("settings.rag");
  const [resources, setResources] = useState<Resource[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const directoryInputRef = useRef<HTMLInputElement>(null);

  const fetchResources = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(resolveServiceURL("rag/resources"), {
        method: "GET",
      });
      if (response.ok) {
        const data = await response.json();
        setResources(data.resources ?? []);
      }
    } catch (error) {
      console.error("Failed to fetch resources:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchResources();
  }, [fetchResources]);

  const processAndUploadFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      if (files.length > MAX_FILES_PER_REQUEST) {
        toast.error(t("tooManyFiles", { max: MAX_FILES_PER_REQUEST }));
        return;
      }
      if (files.some((f) => f.size === 0)) {
        toast.error(t("emptyFile"));
        return;
      }
      const invalid = files.find((f) => !EXT_RE.test(f.name));
      if (invalid) {
        toast.error(t("invalidFileType", { name: invalid.name }));
        return;
      }
      setUploading(true);
      setUploadingCount(files.length);
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }
      try {
        const response = await fetch(resolveServiceURL("rag/upload"), {
          method: "POST",
          body: formData,
        });
        if (response.ok) {
          const data = (await response.json()) as {
            successes?: { file_id: string; resource: { uri: string; title: string; description?: string } }[];
            failed?: { file_id: string; stage: string; reason: string }[];
          };
          const successCount = data.successes?.length ?? 0;
          const failedList = data.failed ?? [];
          const failCount = failedList.length;
          if (failCount > 0) {
            const stageKey: Record<string, string> = {
              parse: "failReasonParse",
              chunk: "failReasonChunk",
              embed: "failReasonEmbed",
              store: "failReasonStore",
            };
            const detailLines = failedList.map(({ file_id, stage, reason }) => {
              const shortReason =
                stageKey[stage] ? t(stageKey[stage] as "failReasonParse") : reason.slice(0, 40) + (reason.length > 40 ? "…" : "");
              return `${file_id}: ${shortReason}`;
            });
            toast.warning(t("uploadPartialSuccess", { success: successCount, failed: failCount }), {
              description: detailLines.join("\n"),
            });
          } else {
            toast.success(
              files.length === 1 ? t("uploadSuccess") : t("uploadSuccessMultiple", { count: files.length }),
            );
          }
          void fetchResources();
        } else {
          const error = await response.json();
          toast.error(error.detail ?? t("uploadFailed"));
        }
      } catch (error) {
        console.error("Upload error:", error);
        toast.error(t("uploadFailed"));
      } finally {
        setUploading(false);
        setUploadingCount(0);
      }
    },
    [t, fetchResources],
  );

  const handleFiles = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = event.target.files;
      if (!fileList || fileList.length === 0) return;
      await processAndUploadFiles(Array.from(fileList));
      event.target.value = "";
    },
    [processAndUploadFiles],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.dataTransfer.types.includes("Files")) return;
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
      if (uploading || loading || !e.dataTransfer.files.length) return;
      try {
        const files = await collectFilesFromDataTransfer(e.dataTransfer);
        if (files.length === 0) {
          toast.error(t("dropNoAllowedFiles"));
          return;
        }
        await processAndUploadFiles(files);
      } catch (err) {
        console.error("Drop parse error:", err);
        toast.error(t("uploadFailed"));
      }
    },
    [uploading, loading, t, processAndUploadFiles],
  );

  return (
    <div
      className={cn(
        "relative flex flex-col gap-4 rounded-lg transition-colors",
        isDragOver && "bg-primary/5 ring-2 ring-primary/30 ring-inset",
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragOver && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-background/90 text-sm font-medium text-primary">
          {t("dropHint")}
        </div>
      )}
      <header>
        <div className="flex items-center justify-between gap-2">
          <h1 className="text-lg font-medium">{t("title")}</h1>
          <div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="sr-only"
              onChange={handleFiles}
              disabled={uploading}
              aria-label={t("uploadFiles")}
            />
            <input
              ref={directoryInputRef}
              type="file"
              multiple
              className="sr-only"
              onChange={handleFiles}
              disabled={uploading}
              aria-label={t("uploadFolder")}
              {...({ webkitdirectory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
            />
            <DropdownMenu>
              <div className="flex">
                <Button
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded-r-none border-r-0"
                >
                  <Upload className="mr-2 h-4 w-4" />
                  {uploading
                ? uploadingCount > 1
                  ? t("uploadingMultiple", { count: uploadingCount })
                  : t("uploading")
                : t("upload")}
                </Button>
                <DropdownMenuTrigger asChild>
                  <Button
                    disabled={uploading}
                    className="rounded-l-none px-2"
                    aria-label={t("uploadFolder")}
                  >
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
              </div>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onSelect={() => {
                    setTimeout(() => fileInputRef.current?.click(), 0);
                  }}
                >
                  <FileText className="mr-2 h-4 w-4" />
                  {t("uploadFiles")}
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    setTimeout(() => directoryInputRef.current?.click(), 0);
                  }}
                >
                  <FolderUp className="mr-2 h-4 w-4" />
                  {t("uploadFolder")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <div className={cn("text-muted-foreground text-sm")}>{t("description")}</div>
      </header>
      <main>
        {loading ? (
          <div className="flex items-center justify-center p-8 text-sm text-gray-500">
            {t("loading")}
          </div>
        ) : resources.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-8 text-center text-gray-500">
            <Database className="h-8 w-8 opacity-50" />
            <p>{t("noResources")}</p>
            <p className="text-xs opacity-90">{t("noResourcesHint")}</p>
          </div>
        ) : (
          <ul className="flex flex-col gap-2">
            {resources.map((resource) => (
              <li
                key={resource.uri}
                className={cn("bg-card flex items-start gap-3 rounded-lg border p-3")}
              >
                <div className={cn("bg-primary/10 rounded p-2")}>
                  <FileText className="text-primary h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="truncate font-medium">{resource.title}</h3>
                  <div className={cn("text-muted-foreground flex items-center gap-2 text-xs")}>
                    <span className="truncate max-w-[300px]" title={resource.uri}>
                      {resource.uri}
                    </span>
                    {resource.description && (
                      <>
                        <span>•</span>
                        <span className="truncate">{resource.description}</span>
                      </>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
};

RAGTab.icon = Database;
RAGTab.displayName = "Resources";
