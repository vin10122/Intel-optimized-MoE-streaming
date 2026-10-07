#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& filepath)
        : filepath_(filepath) {
#ifdef _WIN32
        handle_ = CreateFileA(
            filepath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(handle_, &fileSize)) {
            CloseHandle(handle_);
            throw std::runtime_error("Cannot get file size: " + filepath);
        }
        size_ = static_cast<size_t>(fileSize.QuadPart);
        
        map_handle_ = CreateFileMappingA(
            handle_,
            nullptr,
            PAGE_READONLY,
            0,
            0,
            nullptr
        );
        if (map_handle_ == nullptr) {
            CloseHandle(handle_);
            throw std::runtime_error("Cannot create file mapping: " + filepath);
        }
        
        data_ = static_cast<uint8_t*>(MapViewOfFile(
            map_handle_,
            FILE_MAP_READ,
            0,
            0,
            size_
        ));
        if (data_ == nullptr) {
            CloseHandle(map_handle_);
            CloseHandle(handle_);
            throw std::runtime_error("Cannot map view of file: " + filepath);
        }
#else
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close(fd_);
            throw std::runtime_error("Cannot get file size: " + filepath);
        }
        size_ = static_cast<size_t>(sb.st_size);
        
        data_ = static_cast<uint8_t*>(mmap(
            nullptr,
            size_,
            PROT_READ,
            MAP_PRIVATE,
            fd_,
            0
        ));
        if (data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Cannot mmap file: " + filepath);
        }
#endif
    }
    
    ~MemoryMappedFile() {
        unmap();
    }
    
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    
    const uint8_t* offset(size_t offset) const {
        if (offset >= size_) {
            throw std::out_of_range("Offset out of range");
        }
        return data_ + offset;
    }
    
    void unmap() {
        if (!mapped_) return;
        mapped_ = false;
        
#ifdef _WIN32
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (map_handle_) {
            CloseHandle(map_handle_);
            map_handle_ = nullptr;
        }
        if (handle_) {
            CloseHandle(handle_);
            handle_ = nullptr;
        }
#else
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
#endif
    }
    
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    
private:
    std::string filepath_;
    size_t size_ = 0;
    uint8_t* data_ = nullptr;
    bool mapped_ = true;
    
#ifdef _WIN32
    HANDLE handle_ = nullptr;
    HANDLE map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};
