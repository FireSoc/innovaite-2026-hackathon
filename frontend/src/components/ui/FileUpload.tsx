"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/cn";
import { Upload, X, FileText, Image as ImageIcon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface FileUploadProps {
  onFilesChange: (files: File[]) => void;
  maxFiles?: number;
  maxSizeMB?: number;
  accept?: Record<string, string[]>;
  className?: string;
}

export function FileUpload({
  onFilesChange,
  maxFiles = 10,
  maxSizeMB = 20,
  accept = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"],
    "image/gif": [".gif"],
    "application/pdf": [".pdf"],
  },
  className,
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const newFiles = [...files, ...acceptedFiles].slice(0, maxFiles);
      setFiles(newFiles);
      onFilesChange(newFiles);
    },
    [files, maxFiles, onFilesChange]
  );

  const removeFile = useCallback(
    (index: number) => {
      const newFiles = files.filter((_, i) => i !== index);
      setFiles(newFiles);
      onFilesChange(newFiles);
    },
    [files, onFilesChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: maxFiles - files.length,
    maxSize: maxSizeMB * 1024 * 1024,
    disabled: files.length >= maxFiles,
  });

  const getFileIcon = (file: File) => {
    if (file.type.startsWith("image/")) return <ImageIcon className="h-5 w-5" />;
    return <FileText className="h-5 w-5" />;
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={cn(
          "relative flex flex-col items-center justify-center gap-3 p-8",
          "border-2 border-dashed rounded-2xl cursor-pointer",
          "transition-all duration-200",
          isDragActive
            ? "border-amber bg-amber/5 scale-[1.01]"
            : files.length >= maxFiles
            ? "border-border bg-surface-dim cursor-not-allowed opacity-60"
            : "border-border hover:border-amber/50 hover:bg-surface-hover"
        )}
      >
        <input {...getInputProps()} />
        <div
          className={cn(
            "flex items-center justify-center h-12 w-12 rounded-xl transition-colors",
            isDragActive ? "bg-amber/10 text-amber" : "bg-surface-dim text-text-muted"
          )}
        >
          <Upload className="h-6 w-6" />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-text">
            {isDragActive
              ? "Drop files here..."
              : files.length >= maxFiles
              ? `Maximum ${maxFiles} files reached`
              : "Drag & drop evidence files, or click to browse"}
          </p>
          <p className="text-xs text-text-muted mt-1">
            JPG, PNG, WebP, GIF, or PDF up to {maxSizeMB}MB each ({files.length}/{maxFiles} files)
          </p>
        </div>
      </div>

      {/* File list */}
      <AnimatePresence mode="popLayout">
        {files.map((file, index) => (
          <motion.div
            key={`${file.name}-${file.lastModified}`}
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-3 p-3 bg-surface border border-border rounded-xl group"
          >
            {/* Thumbnail or icon */}
            {file.type.startsWith("image/") ? (
              <div className="h-10 w-10 rounded-lg bg-surface-dim overflow-hidden shrink-0">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="h-full w-full object-cover"
                />
              </div>
            ) : (
              <div className="h-10 w-10 rounded-lg bg-surface-dim flex items-center justify-center text-text-muted shrink-0">
                {getFileIcon(file)}
              </div>
            )}

            {/* File info */}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text truncate">
                {file.name}
              </p>
              <p className="text-xs text-text-muted">{formatSize(file.size)}</p>
            </div>

            {/* Remove button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeFile(index);
              }}
              className="shrink-0 h-8 w-8 flex items-center justify-center rounded-lg text-text-muted hover:text-danger hover:bg-danger-light transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100 focus-ring cursor-pointer"
              aria-label={`Remove ${file.name}`}
            >
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
